# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Directional context transformer for refining expectation tokens."""
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.layers import Dense


class DirectionalContextTransformer(keras.Model):
  """Refines expectation tokens via cross-attention over context tokens."""

  def __init__(self, hidden_dim=64):
    super(DirectionalContextTransformer, self).__init__()
    self.hidden_dim = hidden_dim
    self.query_proj = Dense(hidden_dim, use_bias=False)
    self.key_proj = Dense(hidden_dim, use_bias=False)
    self.value_proj = Dense(hidden_dim, use_bias=False)
    self.output_proj = Dense(3, use_bias=False)

  def call(self, expectation_tokens, context_tokens, training=False):
    """Run cross-attention to refine expectation tokens.

    Args:
      expectation_tokens: [BATCH, N, 3] expectation tokens to be refined.
      context_tokens: [BATCH, M, C] context tokens from decoder feature maps.
      training: (bool) if the training mode is on.

    Returns:
      [BATCH, N, 3] refined expectation tokens.
    """
    if context_tokens is None:
      return expectation_tokens
    queries = self.query_proj(expectation_tokens)
    keys = self.key_proj(context_tokens)
    values = self.value_proj(context_tokens)
    scale = tf.sqrt(tf.cast(self.hidden_dim, tf.float32))
    logits = tf.matmul(queries, keys, transpose_b=True) / scale
    weights = tf.nn.softmax(logits, axis=-1)
    attended = tf.matmul(weights, values)
    update = self.output_proj(attended)
    return expectation_tokens + update
