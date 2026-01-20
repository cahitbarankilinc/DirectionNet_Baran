# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""DirectionNet Architecture."""
from pano_utils import geometry
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import GlobalAveragePooling2D
from tensorflow.compat.v1.keras.layers import LeakyReLU
from tensorflow.compat.v1.keras.layers import UpSampling2D
from tensorflow.compat.v1.keras.models import Sequential


class BottleneckResidualUnit(keras.Model):
  """The fundamental module used to implement deep residual network.

  This module implement the bottleneck residual unit introduced in "Identity
  Mappings in Deep Residual Networks" (https://arxiv.org/pdf/1603.05027.pdf).
  """
  expansion = 2

  def __init__(self,
               n_filters,
               strides=1,
               downsample=None,
               regularization=0.01):
    """Initialize the BottleneckResidualUnit module.

    Args:
      n_filters: (int) the number of output filters.
      strides: (int)  the strides of the convolution.
      downsample: a function to down-sample the feature maps.
      regularization: L2 regularization factor for layer weights.
    """
    super(BottleneckResidualUnit, self).__init__()
    self.bn1 = BatchNormalization()
    self.conv1 = Conv2D(
        n_filters,
        1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))
    self.bn2 = BatchNormalization()
    self.conv2 = Conv2D(
        n_filters,
        3,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))
    self.bn3 = BatchNormalization()
    self.conv3 = Conv2D(
        n_filters*self.expansion,
        1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization))
    self.leaky_relu = LeakyReLU()
    self.downsample = downsample

  def call(self, x, training=False):
    """Call the forward pass of the network.

    Args:
      x: [BATCH, HEIGHT, WIDTH, CHANNELS] the input tensor.
      training: (bool) if the training mode is on.

    Returns:
     [BATCH, HEIGHT, WIDTH, 2*CHANNELS] the output tensor.
    """
    residual = x
    y = self.bn1(x)
    y = self.leaky_relu(y)

    y = self.conv1(y)
    y = self.bn2(y)
    y = self.leaky_relu(y)

    y = self.conv2(y)
    y = self.bn3(y)
    y = self.leaky_relu(y)

    y = self.conv3(y)

    if self.downsample is not None:
      residual = self.downsample(residual)
    return y+residual


class AxialAttention(keras.layers.Layer):
  """Axial attention layer that attends along a single spatial axis."""

  def __init__(self, attention_axis, num_heads=4, regularization=0.01):
    super(AxialAttention, self).__init__()
    if attention_axis not in ('height', 'width'):
      raise ValueError('attention_axis must be "height" or "width".')
    self.attention_axis = attention_axis
    self.num_heads = num_heads
    self.regularization = regularization
    self.qkv = None
    self.proj = None
    self.head_dim = None
    self.scale = None

  def build(self, input_shape):
    channels = int(input_shape[-1])
    if channels % self.num_heads != 0:
      raise ValueError('channels must be divisible by num_heads.')
    self.head_dim = channels // self.num_heads
    self.scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
    self.qkv = Dense(
        3 * channels,
        use_bias=False,
        kernel_regularizer=regularizers.l2(self.regularization))
    self.proj = Dense(
        channels,
        use_bias=False,
        kernel_regularizer=regularizers.l2(self.regularization))
    super(AxialAttention, self).build(input_shape)

  def _split_heads(self, x):
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])
    return tf.transpose(x, [0, 2, 1, 3])

  def _combine_heads(self, x):
    x = tf.transpose(x, [0, 2, 1, 3])
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    return tf.reshape(x, [batch_size, seq_len, self.num_heads * self.head_dim])

  def _scaled_dot_attention(self, x):
    qkv = self.qkv(x)
    q, k, v = tf.split(qkv, 3, axis=-1)
    q = self._split_heads(q)
    k = self._split_heads(k)
    v = self._split_heads(v)
    scores = tf.matmul(q, k, transpose_b=True) / self.scale
    weights = tf.nn.softmax(scores, axis=-1)
    attended = tf.matmul(weights, v)
    attended = self._combine_heads(attended)
    return self.proj(attended)

  def call(self, x, training=False):
    if self.attention_axis == 'height':
      x = tf.transpose(x, [0, 2, 1, 3])
      batch_size = tf.shape(x)[0]
      width = tf.shape(x)[1]
      height = tf.shape(x)[2]
      channels = tf.shape(x)[3]
      x = tf.reshape(x, [batch_size * width, height, channels])
      x = self._scaled_dot_attention(x)
      x = tf.reshape(x, [batch_size, width, height, channels])
      return tf.transpose(x, [0, 2, 1, 3])
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = tf.shape(x)[3]
    x = tf.reshape(x, [batch_size * height, width, channels])
    x = self._scaled_dot_attention(x)
    return tf.reshape(x, [batch_size, height, width, channels])


class DirectionNet(keras.Model):
  """DirectionNet generates spherical probability distributions from images."""

  def __init__(self, n_out, regularization=0.01):
    """Initialize the DirectionNet.

    Args:
      n_out: (int) the number of output distributions.
      regularization: L2 regularization factor for layer weights.
    """
    super(DirectionNet, self).__init__()
    self.encoder = SiameseEncoder()
    self.inplanes = self.encoder.inplanes
    self.decoder_block1 = Sequential([
        Conv2D(256,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 128, regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block2 = Sequential([
        Conv2D(128,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 64, regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block3 = Sequential([
        Conv2D(64,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 32, regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block4 = Sequential([
        Conv2D(32,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 16, regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block5 = Sequential([
        Conv2D(16,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 8, regularization=regularization),
        AxialAttention('height', regularization=regularization),
        AxialAttention('width', regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block6 = Sequential([
        Conv2D(8,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 4, regularization=regularization),
        BatchNormalization(),
        LeakyReLU()])
    self.down_channel = Conv2D(
        n_out, 1, kernel_regularizer=regularizers.l2(regularization))

  def _make_resblock(self, n_blocks, n_filters, strides=1, regularization=0.01):
    """Build Residual blocks from BottleneckResidualUnit layers.

    Args:
      n_blocks: [BATCH, HEIGHT, WIDTH, 3] input source images.
      n_filters: (int) the number of filters.
      strides: (int)  the strides of the convolution.
      regularization: (float) l2 regularization coefficient.

    Returns:
     [BATCH, 1, 1, 1024] image embeddings.
    """
    layers = []
    if strides != 1 or self.inplanes != n_filters*BottleneckResidualUnit.expansion:
      downsample = Conv2D(n_filters*BottleneckResidualUnit.expansion,
                          1,
                          strides=strides,
                          padding='same',
                          use_bias=False)
    else:
      downsample = None
    self.inplanes = n_filters*BottleneckResidualUnit.expansion
    layers.append(BottleneckResidualUnit(
        n_filters, strides, downsample, regularization=regularization))
    for _ in range(1, n_blocks):
      layers.append(BottleneckResidualUnit(
          n_filters, 1, regularization=regularization))
    return Sequential(layers)

  def _spherical_upsampling(self, x):
    """Apply spherical paddings to the feature maps before bilinear upsampling.

    Args:
      x: [BATCH, HEIGHT, WIDTH, CHANNELS] input tensor.

    Returns:
      a tensor [BATCH, 2*(HEIGHT+2), 2*(WIDTH+2), CHANNELS].
    """
    return UpSampling2D(interpolation='bilinear')(
        geometry.equirectangular_padding(x, [[1, 1], [1, 1]]))

  def call(self, img1, img2, training=False):
    """Call the forward pass of the network.

    Args:
      img1: [BATCH, HEIGHT, WIDTH, 3] input source images.
      img2: [BATCH, HEIGHT, WIDTH, 3] input target images.
      training: (bool) if the training mode is on.

    Returns:
     [BATCH, 64, 64, N] N spherical distributions in 64x64 equirectangular grid.
    """
    y = self.encoder(img1, img2, training)

    y = self._spherical_upsampling(y)
    y = self.decoder_block1(y)[:, 1:-1, 1:-1, :]

    y = self._spherical_upsampling(y)
    y = self.decoder_block2(y)[:, 1:-1, 1:-1, :]

    y = self._spherical_upsampling(y)
    y = self.decoder_block3(y)[:, 1:-1, 1:-1, :]

    y = self._spherical_upsampling(y)
    y = self.decoder_block4(y)[:, 1:-1, 1:-1, :]

    y = self._spherical_upsampling(y)
    y = self.decoder_block5(y)[:, 1:-1, 1:-1, :]

    y = self._spherical_upsampling(y)
    y = self.decoder_block6(y)[:, 1:-1, 1:-1, :]

    return self.down_channel(y)


class SiameseEncoder(keras.Model):
  """A Siamese CNN network that encodes stereo images into embeddings."""

  def __init__(self, regularization=0.01):
    super(SiameseEncoder, self).__init__()
    self.inplanes = 64
    # Siamese branch.
    self.siamese = Sequential([
        Conv2D(
            64,
            7,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 128, strides=2, regularization=regularization),
        self._make_resblock(2, 128, strides=2, regularization=regularization),
        self._make_resblock(2, 256, strides=2, regularization=regularization),
    ])
    # Merged main branch.
    self.mainstream = Sequential([
        self._make_resblock(2, 256, strides=2, regularization=regularization),
        self._make_resblock(2, 256, strides=2, regularization=regularization),
    ])
    self.bn = BatchNormalization()
    self.leaky_relu = LeakyReLU()

  def _make_resblock(self, n_blocks, n_filters, strides=1, regularization=0.01):
    """Build Residual blocks from BottleneckResidualUnit layers.

    Args:
      n_blocks: [BATCH, HEIGHT, WIDTH, 3] input source images.
      n_filters: (int) the number of filters.
      strides: (int)  the strides of the convolution.
      regularization: (float) l2 regularization coefficient.

    Returns:
     [BATCH, 1, 1, 1024] image embeddings.
    """
    layers = []
    if strides != 1 or self.inplanes != n_filters*BottleneckResidualUnit.expansion:
      downsample = Conv2D(n_filters*BottleneckResidualUnit.expansion,
                          1,
                          strides=strides,
                          padding='same',
                          use_bias=False)
    else:
      downsample = None
    self.inplanes = n_filters*BottleneckResidualUnit.expansion
    layers.append(BottleneckResidualUnit(
        n_filters, strides, downsample, regularization=regularization))
    for _ in range(1, n_blocks):
      layers.append(BottleneckResidualUnit(
          n_filters, 1, regularization=regularization))
    return Sequential(layers)

  def call(self, img1, img2, training=False):
    """Call the forward pass of the network.

    Args:
      img1: [BATCH, HEIGHT, WIDTH, 3] input source images.
      img2: [BATCH, HEIGHT, WIDTH, 3] input target images.
      training: (bool) if the training mode is on.

    Returns:
     [BATCH, 1, 1, 1024] image embeddings.
    """
    y1 = self.siamese(img1)
    y2 = self.siamese(img2)
    y = self.mainstream(tf.concat([y1, y2], -1))
    y = self.leaky_relu(self.bn(y))
    y = GlobalAveragePooling2D()(y)[:, tf.newaxis, tf.newaxis]
    return y
