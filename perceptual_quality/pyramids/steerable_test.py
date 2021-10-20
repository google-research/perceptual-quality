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
"""Tests for steerable pyramid."""

from absl.testing import parameterized
from perceptual_quality.pyramids import steerable
import tensorflow as tf


class SteerableTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(-1, 1)
  def test_invalid_num_levels_fails(self, num_levels):
    with self.assertRaises(ValueError):
      steerable.SteerablePyramid(num_levels=num_levels)

  @parameterized.parameters(-1, 3)
  def test_invalid_num_subbands_fails(self, num_subbands):
    with self.assertRaises(ValueError):
      steerable.SteerablePyramid(num_subbands=num_subbands)

  def test_invalid_data_format_fails(self):
    with self.assertRaises(ValueError):
      steerable.SteerablePyramid(data_format=3)

  @parameterized.parameters("channels_first", "channels_last")
  def test_invalid_shape_fails(self, data_format):
    pyramid = steerable.SteerablePyramid(data_format=data_format)
    with self.assertRaises(ValueError):
      pyramid(tf.zeros([16]))

  @parameterized.parameters(2, 3, 4)
  def test_number_and_shape_of_scales_match_channels_first(self, num_levels):
    pyramid = steerable.SteerablePyramid(
        num_levels=num_levels, data_format="channels_first")
    image = tf.zeros((3, 32, 16))
    subbands = pyramid(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = {
        2: [(3, 32, 16), (3, 32, 16)],
        3: [(3, 32, 16), (18, 32, 16), (3, 16, 8)],
        4: [(3, 32, 16), (18, 32, 16), (18, 16, 8), (3, 8, 4)],
    }[num_levels]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(2, 3)
  def test_number_and_shape_of_scales_match_channels_last(self, num_levels):
    pyramid = steerable.SteerablePyramid(
        num_levels=num_levels, data_format="channels_last")
    image = tf.zeros((1, 16, 16, 2))
    subbands = pyramid(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = {
        2: [(1, 16, 16, 2), (1, 16, 16, 2)],
        3: [(1, 16, 16, 2), (1, 16, 16, 12), (1, 8, 8, 2)],
    }[num_levels]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(1, 2)
  def test_number_and_shape_of_scales_match_skip_highpass(self, num_levels):
    pyramid = steerable.SteerablePyramid(
        num_levels=num_levels, skip_highpass=True, data_format="channels_last")
    image = tf.zeros((1, 16, 16, 2))
    subbands = pyramid(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = {
        1: [(1, 16, 16, 2)],
        2: [(1, 16, 16, 12), (1, 8, 8, 2)],
    }[num_levels]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(1, 2, 4, 6)
  def test_number_and_shape_of_scales_match_valid(self, num_subbands):
    pyramid = steerable.SteerablePyramid(
        num_levels=4, num_subbands=num_subbands, padding="valid",
        data_format="channels_last")
    image = tf.zeros((72, 64))
    subbands = pyramid(image)
    self.assertLen(subbands, 4)
    expected_shapes = {
        1: [(64, 56, 1), (58, 50, 1), (19, 15, 1), (8, 6, 1)],
        2: [(64, 56, 1), (56, 48, 2), (16, 12, 2), (4, 2, 1)],
        4: [(58, 50, 1), (56, 48, 4), (16, 12, 4), (4, 2, 1)],
        6: [(64, 56, 1), (62, 54, 6), (24, 20, 6), (11, 9, 1)],
    }[num_subbands]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)


if __name__ == "__main__":
  tf.test.main()
