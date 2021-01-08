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

from perceptual_quality.nlpd import distance
import tensorflow.compat.v2 as tf


class DistanceTest(tf.test.TestCase):

  def test_lp_norm_fails_for_unequal_length(self):
    tensors = [tf.zeros((1, 2**i, 2**i, 1)) for i in range(5, 2, -1)]
    with self.assertRaises(ValueError):
      distance.lp_norm(tensors, tensors[:-1])

  def test_lp_norm_shape_is_correct_channels_first(self):
    tensors = [tf.zeros((3, 2, 2**i, 2**i)) for i in range(4, 2, -1)]
    norm = distance.lp_norm(tensors, tensors, spatial_axes=(-2, -1))
    self.assertEqual(norm.shape, (3, 2))

  def test_lp_norm_shape_is_correct_channels_last(self):
    tensors = [tf.zeros((1, 2**i, 2**i, 3)) for i in range(4, 2, -1)]
    norm = distance.lp_norm(tensors, tensors, spatial_axes=(-3, -2))
    self.assertEqual(norm.shape, (1, 3))

  def test_nlpd_fails_for_invalid_dynamic_range(self):
    image = tf.zeros((32, 32, 1))
    with self.assertRaises(ValueError):
      distance.nlpd(image, image, num_levels=3, cdm_min=50, cdm_max=10)
    with self.assertRaises(ValueError):
      distance.nlpd(image, image, num_levels=3, cdm_min=-50, cdm_max=10)

  def test_nlpd_is_zero_for_same_image(self):
    image = tf.random.uniform((32, 32), minval=0, maxval=256, dtype=tf.int32)
    image = tf.cast(image, tf.uint8)
    nlpd = distance.nlpd(
        image, image, num_levels=3, data_format="channels_first")
    self.assertEqual(nlpd, 0.)

  def test_nlpd_fast_is_zero_for_same_image(self):
    image = tf.random.uniform((32, 32), minval=0, maxval=256, dtype=tf.int32)
    image = tf.cast(image, tf.uint8)
    nlpd = distance.nlpd_fast(
        image, image, num_levels=3, data_format="channels_first")
    self.assertEqual(nlpd, 0.)


if __name__ == "__main__":
  tf.test.main()
