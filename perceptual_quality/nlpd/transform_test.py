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
"""Tests for NLP transform."""

from absl.testing import parameterized
from perceptual_quality.nlpd import transform
import tensorflow.compat.v2 as tf


class TransformTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(-1, 0)
  def test_invalid_num_levels_fails(self, num_levels):
    with self.assertRaises(ValueError):
      transform.NLP(num_levels=num_levels)

  def test_invalid_gamma_fails(self):
    with self.assertRaises(ValueError):
      transform.NLP(gamma=-1)

  def test_invalid_data_format_fails(self):
    with self.assertRaises(ValueError):
      transform.NLP(data_format=3)

  def test_invalid_shapes_fail_channels_first(self):
    nlp = transform.NLP(data_format="channels_first")
    with self.assertRaises(ValueError):
      nlp.build((16,))

  def test_invalid_shapes_fail_channels_last(self):
    nlp = transform.NLP(data_format="channels_last")
    with self.assertRaises(ValueError):
      nlp.build((16, 16))

  @parameterized.parameters(1, 2, 3)
  def test_number_and_shape_of_scales_match_channels_first(self, num_levels):
    nlp = transform.NLP(num_levels=num_levels, data_format="channels_first")
    image = tf.zeros((32, 16))
    subbands = nlp(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = [(32, 16), (16, 8), (8, 4)]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)

  @parameterized.parameters(1, 2)
  def test_number_and_shape_of_scales_match_channels_last(self, num_levels):
    nlp = transform.NLP(num_levels=num_levels, data_format="channels_last")
    image = tf.zeros((1, 16, 16, 2))
    subbands = nlp(image)
    self.assertLen(subbands, num_levels)
    expected_shapes = [(1, 16, 16, 2), (1, 8, 8, 2)]
    for subband, shape in zip(subbands, expected_shapes):
      self.assertEqual(subband.shape, shape)


if __name__ == "__main__":
  tf.test.main()
