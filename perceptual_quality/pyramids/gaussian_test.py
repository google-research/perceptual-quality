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
"""Tests for Gaussian pyramid."""

from absl.testing import parameterized
from perceptual_quality.pyramids import gaussian
import tensorflow as tf


class GaussianTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(-1, 0)
  def test_invalid_num_levels_fails(self, num_levels):
    with self.assertRaises(ValueError):
      gaussian.GaussianPyramid(num_levels=num_levels)

  def test_invalid_data_format_fails(self):
    with self.assertRaises(ValueError):
      gaussian.GaussianPyramid(data_format=3)

  @parameterized.parameters("channels_first", "channels_last")
  def test_invalid_shape_fails(self, data_format):
    pyramid = gaussian.GaussianPyramid(data_format=data_format)
    with self.assertRaises(ValueError):
      pyramid(tf.zeros([16]))

  @parameterized.parameters(1, 2, 3)
  def test_number_and_shape_of_scales_match_channels_first(self, num_levels):
    pyramid = gaussian.GaussianPyramid(
        num_levels=num_levels, data_format="channels_first")
    image = tf.zeros((3, 32, 16))
    subbands = pyramid(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = [(3, 32, 16), (3, 16, 8), (3, 8, 4)]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(1, 2)
  def test_number_and_shape_of_scales_match_channels_last(self, num_levels):
    pyramid = gaussian.GaussianPyramid(
        num_levels=num_levels, data_format="channels_last")
    image = tf.zeros((1, 16, 16, 2))
    subbands = pyramid(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = [(1, 16, 16, 2), (1, 8, 8, 2)]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(1, 2, 3)
  def test_number_and_shape_of_scales_match_valid(self, num_levels):
    pyramid = gaussian.GaussianPyramid(
        num_levels=num_levels, padding="valid", data_format="channels_last")
    image = tf.zeros((48, 64))
    subbands = pyramid(image)
    expected_shapes = {
        1: [(48, 64)],
        2: [(48, 64), (22, 30)],
        3: [(48, 64), (22, 30), (9, 13)],
    }[num_levels]
    self.assertLen(subbands, num_levels)
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)


if __name__ == "__main__":
  tf.test.main()
