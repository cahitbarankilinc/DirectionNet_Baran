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

"""Directional context transformer layer."""
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.layers import Attention
from tensorflow.compat.v1.keras.layers import Dense
from tensorflow.compat.v1.keras.layers import Dropout
from tensorflow.compat.v1.keras.layers import LayerNormalization


class DirectionalContextTransformer(keras.Model):
  """A lightweight transformer-style block for directional context."""

  def __init__(self, token_dim, ff_dim=128, dropout_rate=0.1):
    super(DirectionalContextTransformer, self).__init__()
    self.attention = Attention()
    self.norm1 = LayerNormalization(epsilon=1e-6)
    self.norm2 = LayerNormalization(epsilon=1e-6)
    self.ffn = keras.Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(token_dim),
    ])
    self.dropout = Dropout(dropout_rate)

  def call(self, tokens, training=False):
    """Apply self-attention and feed-forward refinement.

    Args:
      tokens: [BATCH, TOKENS, DIM] token embeddings.
      training: (bool) if the training mode is on.

    Returns:
      [BATCH, TOKENS, DIM] refined token embeddings.
    """
    attn_out = self.attention([tokens, tokens])
    attn_out = self.dropout(attn_out, training=training)
    x = self.norm1(tokens + attn_out)
    ffn_out = self.ffn(x)
    ffn_out = self.dropout(ffn_out, training=training)
    return self.norm2(x + ffn_out)
