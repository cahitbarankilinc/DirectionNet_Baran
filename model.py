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
from tensorflow.compat.v1.keras.layers import GlobalAveragePooling2D
from tensorflow.compat.v1.keras.layers import LeakyReLU
from tensorflow.compat.v1.keras.layers import MultiHeadAttention
from tensorflow.compat.v1.keras.layers import UpSampling2D
from tensorflow.compat.v1.keras.models import Sequential


class WindowAttentionBlock(keras.layers.Layer):
  """Windowed multi-head self-attention block (Swin-style local attention)."""

  def __init__(self, num_heads, key_dim, window_size=4, shift_size=0, **kwargs):
    super(WindowAttentionBlock, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.window_size = window_size
    self.shift_size = shift_size
    self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

  def call(self, x, training=False):
    input_shape = tf.shape(x)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    window_size = self.window_size
    shift_size = self.shift_size

    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    padded = tf.pad(
        x,
        [[0, 0], [0, pad_height], [0, pad_width], [0, 0]],
        constant_values=0)
    if shift_size:
      padded = tf.roll(padded, shift=[-shift_size, -shift_size], axis=[1, 2])

    padded_shape = tf.shape(padded)
    padded_height = padded_shape[1]
    padded_width = padded_shape[2]
    windows_height = padded_height // window_size
    windows_width = padded_width // window_size

    reshaped = tf.reshape(
        padded,
        [batch_size, windows_height, window_size, windows_width, window_size,
         channels])
    windows = tf.transpose(reshaped, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [-1, window_size * window_size, channels])

    attended = self.attention(windows, windows, training=training)
    attended = tf.reshape(
        attended,
        [batch_size, windows_height, windows_width, window_size, window_size,
         channels])
    attended = tf.transpose(attended, [0, 1, 3, 2, 4, 5])
    attended = tf.reshape(
        attended, [batch_size, padded_height, padded_width, channels])

    if shift_size:
      attended = tf.roll(attended, shift=[shift_size, shift_size], axis=[1, 2])

    if pad_height or pad_width:
      attended = attended[:, :height, :width, :]
    return x + attended


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
        WindowAttentionBlock(num_heads=4, key_dim=8, window_size=4),
        BatchNormalization(),
        LeakyReLU()])
    self.decoder_block5 = Sequential([
        Conv2D(16,
               3,
               use_bias=False,
               kernel_regularizer=regularizers.l2(regularization)),
        self._make_resblock(2, 8, regularization=regularization),
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
