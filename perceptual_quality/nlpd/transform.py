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

from perceptual_quality.pyramids import laplacian
import tensorflow as tf


class NLP(laplacian.LaplacianPyramid):
  """Normalized Laplacian pyramid transformation.

  This implements the perceptual transform f(S) as defined in the paper:

  > "Perceptually optimized image rendering"</br>
  > V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
  > https://doi.org/10.1364/JOSAA.34.001511

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

  Inputs to this transform are expected to be linear luminances in cd∕m^2. To
  convert from typical sRGB pixel values stored in an `image` array to linear
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
    super().__init__(num_levels=num_levels, data_format=data_format, name=name)
    if gamma is not None and gamma <= 0:
      raise ValueError(f"gamma must be either None or a positive number, "
                       f"received {gamma}.")
    self.gamma = None if gamma is None else float(gamma)

  def build(self, input_shape):
    super().build(input_shape)
    # We name these variables as in the paper.
    # pylint:disable=invalid-name

    # 5-tap lowpass filter for Laplacian pyramid.
    L = tf.constant([.05, .25, .4, .25, .05], dtype=self.dtype)
    self.L = L[None, :, None, None] * L[:, None, None, None]

    # Normalization pool for bandpass scales.
    P = tf.constant([
        [4e-2, 4e-2, 5e-2, 4e-2, 4e-2],
        [4e-2, 3e-2, 4e-2, 3e-2, 4e-2],
        [5e-2, 4e-2, 5e-2, 4e-2, 5e-2],
        [4e-2, 3e-2, 4e-2, 3e-2, 4e-2],
        [4e-2, 4e-2, 5e-2, 4e-2, 4e-2],
    ], dtype=self.dtype)
    self.P = tf.reshape(P, (5, 5, 1, 1))

    # Additive constants for the normalization.
    self.sigma = tf.constant(.17, dtype=self.dtype)
    self.sigma_l = tf.constant(4.86, dtype=self.dtype)

    # pylint:enable=invalid-name

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

    kernel = tf.broadcast_to(self.L, (5, 5, num_channels, 1))
    kernel4 = tf.broadcast_to(4. * self.L, (5, 5, num_channels, 1))
    kernelp = tf.broadcast_to(self.P, (5, 5, num_channels, 1))

    if self.gamma is not None:
      x **= self.gamma

    subbands = []
    for _ in range(self.num_levels - 1):
      z, x = self._laplacian_level(x, kernel, kernel4)
      pool = tf.nn.depthwise_conv2d(
          abs(z), kernelp, strides=(1, 1, 1, 1), padding="SAME",
          data_format=self._data_format)
      subbands.append(z / (pool + self.sigma))
    subbands.append(x / (abs(x) + self.sigma_l))

    if rank == 2:
      subbands = [tf.squeeze(s, (0, channel_axis)) for s in subbands]
    elif rank == 3:
      subbands = [tf.squeeze(s, 0) for s in subbands]

    return subbands
