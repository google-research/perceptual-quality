# Copyright 2021 Google LLC. All Rights Reserved.
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
# ==============================================================================
"""Gaussian pyramid."""

import tensorflow as tf


class GaussianPyramid(tf.keras.layers.Layer):
  """Gaussian pyramid transformation.

  The transform operates only across spatial dimensions. With
  `data_format == 'channels_first'`, the input must have shape `(H, W)`,
  `(C, H, W)`, or `(N, C, H, W)`. With `data_format == 'channels_last'`, the
  input must have shape `(H, W)`, `(H, W, C)`, or `(N, H, W, C)`. There are
  `num_level` output tensors with varying spatial dimensions. The non-spatial
  dimensions are simply passed through to the outputs. For example:

  ```
  data_format == 'channels_last':
  (N, H, W, C) -> [(N, H1, W1, C), (N, H2, W2, C), ...]
  (H, W, C) -> [(H1, W1, C), (H2, W2, C), ...]

  data_format == 'channels_first':
  (N, C, H, W) -> [(N, C, H1, W1), (N, C, H2, W2), ...]
  (H, W) -> [(H1, W1), (H2, W2), ...]
  ```
  """

  def __init__(self, num_levels=6, padding="same_reflect",
               data_format="channels_last", name="GaussianPyramid"):
    """Initializer.

    Args:
      num_levels: Integer. The number of pyramid levels.
      padding: String. `'same_reflect'` or `'valid'`.
      data_format: String. Either `'channels_first'` or `'channels_last'`. The
        data format used for the convolution operations.
      name: String. A name for the transformation.
    """
    super().__init__(name=name)
    if num_levels < 1:
      raise ValueError(f"Must have at least one level, received {num_levels}.")
    if padding not in ("same_reflect", "valid"):
      raise ValueError(
          f"padding must be either 'same_reflect' or 'valid', received "
          f"'{data_format}'.")
    if data_format not in ("channels_first", "channels_last"):
      raise ValueError(
          f"data_format must be either 'channels_first' or 'channels_last', "
          f"received '{data_format}'.")
    self.num_levels = int(num_levels)
    self.padding = str(padding)
    self.data_format = str(data_format)

  @property
  def _data_format(self):
    return {"channels_first": "NCHW", "channels_last": "NHWC"}[self.data_format]

  def _pad_tuple(self, unpadded, value):
    if self.data_format == "channels_first":
      return (value, value) + unpadded
    else:
      return (value,) + unpadded + (value,)

  def call(self, image):
    x = tf.convert_to_tensor(image, dtype=self.dtype)
    rank = x.shape.rank
    channel_axis = self._data_format.find("C")
    if not 2 <= rank <= 4:
      raise ValueError(
          f"Input image tensor must be rank 2, 3, or 4, got shape {x.shape}.")
    if rank in (2, 3):
      x = tf.expand_dims(x, 0)
    if rank == 2:
      x = tf.expand_dims(x, channel_axis)
    num_channels = x.shape[channel_axis]

    kernel = tf.constant([.05, .25, .4, .25, .05], dtype=self.dtype)
    kernel = kernel[None, :, None, None] * kernel[:, None, None, None]
    kernel = tf.broadcast_to(kernel, (5, 5, num_channels, 1))

    subbands = [x]
    for _ in range(self.num_levels - 1):
      if self.padding == "same_reflect":
        x = tf.pad(x, paddings=self._pad_tuple(((2, 2), (2, 2)), (0, 0)),
                   mode="REFLECT")
      else:
        assert self.padding == "valid", self.padding
      x = tf.nn.depthwise_conv2d(
          x, kernel, strides=self._pad_tuple((2, 2), 1), padding="VALID",
          data_format=self._data_format)
      subbands.append(x)

    if rank == 2:
      return [tf.squeeze(s, (0, channel_axis)) for s in subbands]
    elif rank == 3:
      return [tf.squeeze(s, 0) for s in subbands]
    else:
      return subbands
