# Copyright 2020 Google LLC. All Rights Reserved.
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
"""NLP transform."""

import tensorflow.compat.v2 as tf


class NLP(tf.keras.layers.Layer):
  """Normalized Laplacian pyramid transformation.

  This implements the perceptual transform f(S) as defined in the paper:

  > "Perceptually optimized image rendering"</br>
  > V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
  > https://doi.org/10.1364/JOSAA.34.001511

  Inputs to this transform are expected to be linear luminances in cd∕m^2. The
  input tensor must have at least two spatial dimensions, and a trailing channel
  dimension if `data_format == 'channels_last'`. Any leading dimensions are
  simply passed through to the output.

  To convert from typical sRGB pixel values stored in an `image` array to linear
  luminances, one could assume a display with a dynamic range between 5 and 180
  cd/m^2, and make the following transformation:

  ```
  linear_image = (image / 255) ** 2.4  # sRGB has a gamma of approximately 2.4
  S = linear_image * 175 + 5  # luminance values
  subbands = NLP()(S)
  ```

  For a quick-and-dirty approximation, we can also turn the power law
  transformation controlled by the `gamma` parameter off, and simply do:

  ```
  subbands = NLP(gamma=None)(image)
  ```
  Note that this equivalent to assuming a display with a dynamic range between
  0 and 255 cd/m^2, and a colorspace with a gamma correction of 2.6, which is
  not typical and may result in suboptimal correlation with mean opinion scores.
  """

  def __init__(self, num_levels=6, gamma=1/2.6, data_format="channels_last",
               name="NLP"):
    """Initializer.

    Args:
      num_levels: Integer. The number of pyramid levels, including the lowpass
        residual.
      gamma: Float or None. The gamma parameter for the power law transformation
        approximating the response of retinal photoreceptors. The default
        (1/2.6) is appropriate for linear luminance inputs. Set to `None` for
        omitting the transformation.
      data_format: String. Either `'channels_first'` or `'channels_last'`.
      name: String. A name for the transformation.
    """
    super().__init__(name=name)
    if num_levels < 1:
      raise ValueError(f"Must have at least one level, received {num_levels}.")
    if gamma is not None and gamma <= 0:
      raise ValueError(f"gamma must be either None or a positive number, "
                       f"received {gamma}.")
    if data_format not in ("channels_first", "channels_last"):
      raise ValueError(
          f"data_format must be either 'channels_first' or 'channels_last', "
          f"received '{data_format}'.")
    self.num_levels = int(num_levels)
    self.gamma = None if gamma is None else float(gamma)
    self.data_format = str(data_format)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == "channels_last":
      if input_shape.rank < 3:
        raise ValueError(f"Input must have at least 3 dimensions, received "
                         f"{input_shape.rank}.")
    else:
      if input_shape.rank < 2:
        raise ValueError(f"Input must have at least 2 dimensions, received "
                         f"{input_shape.rank}.")
    super().build(input_shape)

    # We name these variables as in the paper.
    # pylint:disable=invalid-name
    # H/W separable five-tap filter for Laplacian pyramid.
    self.L = tf.constant([.05, .25, .4, .25, .05], dtype=self.dtype)

    # Normalization pool for bandpass scales.
    P = tf.constant([
        [4e-2, 4e-2, 5e-2, 4e-2, 4e-2],
        [4e-2, 3e-2, 4e-2, 3e-2, 4e-2],
        [5e-2, 4e-2, 5e-2, 4e-2, 5e-2],
        [4e-2, 3e-2, 4e-2, 3e-2, 4e-2],
        [4e-2, 4e-2, 5e-2, 4e-2, 4e-2],
    ], dtype=self.dtype)
    self.P = tf.reshape(P, (5, 5, 1, 1))
    # pylint:enable=invalid-name

    # Additive constants for the normalization.
    self.sigma = tf.constant(.17, dtype=self.dtype)
    self.sigma_l = tf.constant(4.86, dtype=self.dtype)

  def _fold_dimensions(self, image):
    """Folds batch/channel/depth dimensions for easier processing."""
    # If not in channels_first format yet, convert to it first.
    if self.data_format == "channels_last":
      rank = tf.rank(image)
      image = tf.transpose(image, [0, rank - 1] + list(range(1, rank - 1)))
    # Fold all but the spatial dimensions, so we can always use 2D convolutions
    # without duplicating filters.
    full_shape = tf.shape(image)
    folded_shape = tf.concat([[-1], full_shape[-2:], [1]], 0)
    return full_shape[:-2], tf.reshape(image, folded_shape)

  def _unfold_dimensions(self, subband, folded_dims):
    """Undoes `_fold_dimensions`."""
    full_shape = tf.concat([folded_dims, tf.shape(subband)[-3:-1]], 0)
    subband = tf.reshape(subband, full_shape)
    if self.data_format == "channels_last":
      rank = tf.rank(subband)
      return tf.transpose(subband, [0] + list(range(2, rank)) + [1])
    else:
      return subband

  def _laplacian_level(self, x, l_w, l_h, l_w2, l_h2):
    """Performs one down/upsampling level of the Laplacian pyramid."""
    x_shape = tf.shape(x)
    padded = tf.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="REFLECT")
    down_h = tf.nn.conv2d(
        padded, l_h, (2, 1), padding="VALID", data_format="NHWC")
    down_hw = tf.nn.conv2d(
        down_h, l_w, (1, 2), padding="VALID", data_format="NHWC")
    up_h_shape = tf.concat([tf.shape(down_hw)[:-2], x_shape[-2:]], 0)
    up_w = tf.nn.conv2d_transpose(
        down_hw, l_w2, up_h_shape, (1, 2),
        padding=((0, 0), (0, 0), (4, 4), (0, 0)), data_format="NHWC")
    up_hw = tf.nn.conv2d_transpose(
        up_w, l_h2, x_shape, (2, 1),
        padding=((0, 0), (4, 4), (0, 0), (0, 0)), data_format="NHWC")
    # For input x^(n), these are called z^(n) and x^(n+1) in the paper.
    return x - up_hw, down_hw[:, 1:-1, 1:-1, :]

  def call(self, image):
    folded_dims, x = self._fold_dimensions(image)

    if self.gamma is not None:
      x **= self.gamma

    l_w = tf.reshape(self.L, (1, 5, 1, 1))
    l_h = tf.reshape(self.L, (5, 1, 1, 1))
    l_w2 = 2. * l_w
    l_h2 = 2. * l_h

    subbands = []
    for _ in range(self.num_levels - 1):
      z, x = self._laplacian_level(x, l_w, l_h, l_w2, l_h2)
      pool = tf.nn.conv2d(abs(z), self.P, 1, padding="SAME", data_format="NHWC")
      subbands.append(z / (pool + self.sigma))
    subbands.append(x / (abs(x) + self.sigma_l))

    return [self._unfold_dimensions(s, folded_dims) for s in subbands]
