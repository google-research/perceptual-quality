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
"""Nested norm and distance function for NLPD."""

from perceptual_quality.nlpd import transform
import tensorflow.compat.v2 as tf


def lp_norm(subbands_a, subbands_b, spatial_axes=(-3, -2)):
  """Normalized Laplacian pyramid distance.

  This implements the nested Lp norm as defined in the paper:

  > "Perceptually optimized image rendering"</br>
  > V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
  > https://doi.org/10.1364/JOSAA.34.001511

  Args:
    subbands_a: Sequence of `Tensor`s. NLP subband decompositions of image A.
    subbands_b: Sequence of `Tensor`s. NLP subband decompositions of image B.
    spatial_axes: Sequence of two integers. Should specify the axes of the
      tensors that contain the spatial dimensions. The default is (-3, -2)
      (for instance, for NHWC format).

  Returns:
    A `Tensor` giving the NLPD values for each of the non-spatial dimensions
    (e.g. shaped NC for NHWC inputs).
  """
  if len(subbands_a) != len(subbands_b):
    raise ValueError("`subbands_a` and `subbands_b` must have equal length.")
  scales = []
  for a, b in zip(subbands_a, subbands_b):
    mse = tf.reduce_mean(tf.math.squared_difference(a, b), axis=spatial_axes)
    scales.append(mse ** (.5 * .6))
  return tf.reduce_mean(tf.stack(scales, axis=0), axis=0) ** (1 / .6)


def nlpd(image_a, image_b, num_levels=6, cdm_min=5, cdm_max=180,
         data_format="channels_last"):
  """Normalized Laplacian pyramid distance.

  This implements the NLPD as defined in the paper:

  > "Perceptually optimized image rendering"</br>
  > V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
  > https://doi.org/10.1364/JOSAA.34.001511

  The inputs are assumed to be sRGB images, and the display is assumed to have a
  dynamic range of `cdm_min` to `cdm_max` cd/m^2.

  Args:
    image_a: `Tensor` containing image A.
    image_b: `Tensor` containing image B.
    num_levels: Integer. The number of pyramid levels, including the lowpass
      residual.
    cdm_min: Float. Minimum assumed cd/m^2 of display.
    cdm_max: Float. Maximum assumed cd/m^2 of display.
    data_format: String. Either `'channels_first'` or `'channels_last'`.

  Returns:
    A `Tensor` giving the NLPD values for each of the non-spatial dimensions
    (e.g. shaped NC for NHWC inputs).
  """
  if not 0 <= cdm_min < cdm_max:
    raise ValueError("Must have `0 <= cdm_min < cdm_max`.")

  if image_a.dtype.is_integer:
    image_a = tf.cast(image_a, tf.float32)
  if image_b.dtype.is_integer:
    image_b = tf.cast(image_b, tf.float32)

  def convert_to_cdm2(image):
    return ((image / 255) ** 2.4) * (cdm_max - cdm_min) + cdm_min

  nlp = transform.NLP(num_levels=num_levels, data_format=data_format)

  subbands_a = nlp(convert_to_cdm2(image_a))
  subbands_b = nlp(convert_to_cdm2(image_b))

  if data_format == "channels_first":
    spatial_axes = (-2, -1)
  else:
    spatial_axes = (-3, -2)
  return lp_norm(subbands_a, subbands_b, spatial_axes=spatial_axes)


def nlpd_fast(image_a, image_b, num_levels=6, data_format="channels_last"):
  """Normalized Laplacian pyramid distance.

  This implements a quick-and-dirty approximation to the NLPD, which is defined
  in the paper:

  > "Perceptually optimized image rendering"</br>
  > V. Laparra, A. Berardino, J. Ballé and E. P. Simoncelli</br>
  > https://doi.org/10.1364/JOSAA.34.001511

  The inputs are assumed to be sRGB images. This approximation omits the
  colorspace conversion.

  Args:
    image_a: `Tensor` containing image A.
    image_b: `Tensor` containing image B.
    num_levels: Integer. The number of pyramid levels, including the lowpass
      residual.
    data_format: String. Either `'channels_first'` or `'channels_last'`.

  Returns:
    A `Tensor` giving the NLPD values for each of the non-spatial dimensions
    (e.g. shaped NC for NHWC inputs).
  """
  if image_a.dtype.is_integer:
    image_a = tf.cast(image_a, tf.float32)
  if image_b.dtype.is_integer:
    image_b = tf.cast(image_b, tf.float32)

  nlp = transform.NLP(
      num_levels=num_levels, gamma=None, data_format=data_format)

  subbands_a = nlp(image_a)
  subbands_b = nlp(image_b)

  if data_format == "channels_first":
    spatial_axes = (-2, -1)
  else:
    spatial_axes = (-3, -2)
  return lp_norm(subbands_a, subbands_b, spatial_axes=spatial_axes)
