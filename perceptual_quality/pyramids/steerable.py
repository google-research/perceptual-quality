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
"""Steerable pyramid."""

from perceptual_quality.third_party.pyrtools import spfilters
import tensorflow as tf


class SteerablePyramid(tf.keras.layers.Layer):
  """Steerable pyramid transformation.

  The transform operates only across spatial dimensions. With
  `data_format == 'channels_first'`, the input must have shape `(H, W)`,
  `(C, H, W)`, or `(N, C, H, W)`. With `data_format == 'channels_last'`, the
  input must have shape `(H, W)`, `(H, W, C)`, or `(N, H, W, C)`. There are
  `num_level` output tensors with varying spatial dimensions. The non-spatial
  dimensions are simply passed through to the outputs.

  For bandpass scales (all scales except the first and last), the channel
  dimension will be multiplied by `num_subbands`. If the input shape is
  `(H, W)`, the outputs will all have a channel dimension added appropriately.

  For example (here, `S = num_subbands`):

  ```
  data_format == 'channels_last':
  (N, H, W, C) -> [(N, H1, W1, C), (N, H2, W2, S*C), ...]
  (H, W, C) -> [(H1, W1, C), (H2, W2, S*C), ...]

  data_format == 'channels_first':
  (N, C, H, W) -> [(N, C, H1, W1), (N, S*C, H2, W2), ...]
  (H, W) -> [(1, H1, W1), (S, H2, W2), ...]
  ```
  """

  def __init__(self, num_levels=6, num_subbands=6, padding="same_zeros",
               data_format="channels_last", name="SteerablePyramid"):
    """Initializer.

    Args:
      num_levels: Integer. The number of pyramid levels, including the lowpass
        and highpass residual.
      num_subbands: Integer. The number of subbands per level. Must be one of
        1, 2, 4, or 6.
      padding: String. `'same_zeros'` or `'valid'`.
      data_format: String. Either `'channels_first'` or `'channels_last'`. The
        data format used for the convolution operations.
      name: String. A name for the transformation.
    """
    super().__init__(name=name)
    if num_levels < 2:
      raise ValueError(f"Must have at least two levels, received {num_levels}.")
    if num_subbands not in (1, 2, 4, 6):
      raise ValueError(f"{num_subbands} subbands per level are not supported.")
    if padding not in ("same_zeros", "valid"):
      raise ValueError(
          f"padding must be either 'same_zeros' or 'valid', received "
          f"'{data_format}'.")
    if data_format not in ("channels_first", "channels_last"):
      raise ValueError(
          f"data_format must be either 'channels_first' or 'channels_last', "
          f"received '{data_format}'.")
    self.num_levels = int(num_levels)
    self.num_subbands = int(num_subbands)
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

  def build(self, input_shape):
    super().build(input_shape)
    filters = getattr(spfilters, f"sp_filters{self.num_subbands-1}")()

    self.lo0filt = tf.constant(
        filters["lo0filt"], dtype=self.dtype)[:, :, None, None]
    self.hi0filt = tf.constant(
        filters["hi0filt"], dtype=self.dtype)[:, :, None, None]
    self.lofilt = tf.constant(
        filters["lofilt"], dtype=self.dtype)[:, :, None, None]

    bfilts = tf.constant(filters["bfilts"], dtype=self.dtype)
    assert bfilts.shape[1] == self.num_subbands, bfilts.shape
    support = int(round(float(bfilts.shape[0]) ** .5))
    bfilts = tf.reshape(bfilts, (support, support, self.num_subbands))
    self.bfilts = bfilts[:, :, None, :]

    # All filters should have odd-length supports.
    assert self.lo0filt.shape[0] % 2 == 1, self.lo0filt.shape
    assert self.lo0filt.shape[1] % 2 == 1, self.lo0filt.shape
    assert self.hi0filt.shape[0] % 2 == 1, self.hi0filt.shape
    assert self.hi0filt.shape[1] % 2 == 1, self.hi0filt.shape
    assert self.lofilt.shape[0] % 2 == 1, self.lofilt.shape
    assert self.lofilt.shape[1] % 2 == 1, self.lofilt.shape
    assert self.bfilts.shape[0] % 2 == 1, self.bfilts.shape
    assert self.bfilts.shape[1] % 2 == 1, self.bfilts.shape

    # Lowpass filters should have length 4*n+1, so that all scales can be
    # pixel-aligned.
    assert (self.lofilt.shape[0] // 2) % 2 == 0, self.lofilt.shape
    assert (self.lofilt.shape[1] // 2) % 2 == 0, self.lofilt.shape

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

    lo0filt = tf.tile(self.lo0filt, (1, 1, num_channels, 1))
    hi0filt = tf.tile(self.hi0filt, (1, 1, num_channels, 1))
    bfilts = tf.tile(self.bfilts, (1, 1, num_channels, 1))
    lofilt = tf.tile(self.lofilt, (1, 1, num_channels, 1))

    padding = self.padding.upper()
    lo_padding = self.padding.upper()
    if self.padding == "same_zeros":
      padding = "SAME"
      # Use explicit padding mode for image-size independent padding.
      lo_shape = self.lofilt.shape.as_list()
      assert lo_shape[0] % 2
      assert lo_shape[1] % 2
      lo_padding = self._pad_tuple((
          (lo_shape[0] // 2, lo_shape[0] // 2),
          (lo_shape[1] // 2, lo_shape[1] // 2),
      ), (0, 0))

    lo = tf.nn.depthwise_conv2d(
        x, lo0filt, strides=(1, 1, 1, 1), padding=padding,
        data_format=self._data_format)
    hi0 = tf.nn.depthwise_conv2d(
        x, hi0filt, strides=(1, 1, 1, 1), padding=padding,
        data_format=self._data_format)
    subbands = [hi0]

    for _ in range(self.num_levels - 2):
      ba = tf.nn.depthwise_conv2d(
          lo, bfilts, strides=(1, 1, 1, 1), padding=padding,
          data_format=self._data_format)
      subbands.append(ba)
      lo = tf.nn.depthwise_conv2d(
          lo, lofilt, strides=self._pad_tuple((2, 2), 1), padding=lo_padding,
          data_format=self._data_format)
    subbands.append(lo)

    if rank in (2, 3):
      return [tf.squeeze(s, 0) for s in subbands]
    else:
      return subbands
