# coding=utf-8
"""Directional transformer module for contextualizing DirectionNet outputs."""

import math

from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras


def _approximate_gelu(x):
  """TF1-friendly GELU approximation used when keras.activations.gelu is absent."""
  coeff = tf.cast(tf.sqrt(tf.constant(2.0 / math.pi, dtype=tf.float32)), x.dtype)
  return 0.5 * x * (1.0 + tf.tanh(coeff * (x + 0.044715 * tf.pow(x, 3))))


class LayerNormalization(keras.layers.Layer):
  """Layer normalization supporting TF1 graph mode."""

  def __init__(self, epsilon=1e-6, **kwargs):
    super(LayerNormalization, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    dim = input_shape[-1]
    if dim is None:
      raise ValueError(
          'The last dimension of the input to LayerNormalization must be set.')
    self.gamma = self.add_weight(
        'gamma',
        shape=[dim],
        initializer=tf.initializers.ones())
    self.beta = self.add_weight(
        'beta',
        shape=[dim],
        initializer=tf.initializers.zeros())
    super(LayerNormalization, self).build(input_shape)

  def call(self, inputs):
    mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
    normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
    return normalized * self.gamma + self.beta


class DirectionalContextTransformer(keras.Model):
  """Lightweight transformer that contextualizes direction tokens."""

  def __init__(self,
               hidden_size=128,
               num_heads=4,
               mlp_dim=256,
               dropout_rate=0.1,
               name='directional_context_transformer'):
    super(DirectionalContextTransformer, self).__init__(name=name)
    if hidden_size % num_heads != 0:
      raise ValueError('hidden_size must be divisible by num_heads.')
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.dropout_rate = dropout_rate

    self.input_projection = keras.layers.Dense(hidden_size)
    self.context_projection = keras.layers.Dense(3)
    self.positional_embedding = self.add_weight(
        'positional_embedding',
        shape=[1, 5, hidden_size],
        initializer=tf.initializers.glorot_uniform())
    self.attention_qkv = keras.layers.Dense(hidden_size * 3, use_bias=False)
    self.attention_output = keras.layers.Dense(hidden_size, use_bias=False)
    self.attention_dropout = keras.layers.Dropout(dropout_rate)
    self.norm1 = LayerNormalization()
    self.norm2 = LayerNormalization()
    # TF1-compat mode may not expose tf.nn.gelu or keras.activations.gelu; use a
    # local approximation to retain the intended nonlinearity consistently.
    self.mlp_dense1 = keras.layers.Dense(mlp_dim, activation=_approximate_gelu)
    self.mlp_dropout1 = keras.layers.Dropout(dropout_rate)
    self.mlp_dense2 = keras.layers.Dense(hidden_size)
    self.mlp_dropout2 = keras.layers.Dropout(dropout_rate)
    self.output_projection = keras.layers.Dense(3)
    # Avoid spamming logs—only announce activation the first time the module
    # participates in the graph.
    self._usage_logged = False

  def _split_heads(self, x):
    batch = tf.shape(x)[0]
    length = tf.shape(x)[1]
    x = tf.reshape(x, [batch, length, self.num_heads, self.head_dim])
    return tf.transpose(x, [0, 2, 1, 3])

  def _merge_heads(self, x):
    batch = tf.shape(x)[0]
    length = tf.shape(x)[2]
    x = tf.transpose(x, [0, 2, 1, 3])
    return tf.reshape(x, [batch, length, self.hidden_size])

  def call(self, expectation, context_embedding, training=False):
    """Contextualize the raw expectation vectors.

    Args:
      expectation: [BATCH, N, 3] tensor of raw expectation vectors with N in {3,
        4}.
      context_embedding: [BATCH, 1024] Siamese encoder bottleneck embedding.
      training: Whether the module is running in training mode.

    Returns:
      [BATCH, N, 3] tensor of refined, unit-normalized direction vectors.
    """
    expectation_length = tf.shape(expectation)[1]
    context_token = self.context_projection(context_embedding)
    context_token = context_token[:, tf.newaxis, :]
    tokens = tf.concat([expectation, context_token], axis=1)

    token_features = self.input_projection(tokens)
    token_features += self.positional_embedding[:, :tf.shape(token_features)[1], :]

    # Multi-head self-attention with pre-norm residual.
    attn_input = self.norm1(token_features)
    qkv = self.attention_qkv(attn_input)
    q, k, v = tf.split(qkv, 3, axis=-1)
    q = self._split_heads(q)
    k = self._split_heads(k)
    v = self._split_heads(v)
    scale = tf.cast(self.head_dim, attn_input.dtype) ** -0.5
    attention_logits = tf.matmul(q, k, transpose_b=True) * scale
    attention_weights = tf.nn.softmax(attention_logits, axis=-1)
    attention_weights = self.attention_dropout(attention_weights, training=training)
    attention_output = tf.matmul(attention_weights, v)
    attention_output = self._merge_heads(attention_output)
    attention_output = self.attention_output(attention_output)
    attention_output = self.attention_dropout(attention_output, training=training)
    token_features += attention_output

    # Feed-forward block with pre-norm residual.
    mlp_input = self.norm2(token_features)
    mlp_output = self.mlp_dense1(mlp_input)
    mlp_output = self.mlp_dropout1(mlp_output, training=training)
    mlp_output = self.mlp_dense2(mlp_output)
    mlp_output = self.mlp_dropout2(mlp_output, training=training)
    token_features += mlp_output

    refined = self.output_projection(token_features[:, :expectation_length, :])
    outputs = tf.nn.l2_normalize(refined, axis=-1)
    if not self._usage_logged:
      try:
        param_count = self.count_params()
      except ValueError:
        param_count = None
      mode = 'training' if training else 'inference'
      if param_count is not None:
        logging.info(
            'Directional transformer active in %s mode with %d trainable '
            'parameters.', mode, param_count)
      else:
        logging.info('Directional transformer active in %s mode.', mode)
      self._usage_logged = True
    return outputs
