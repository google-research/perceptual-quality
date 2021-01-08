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
"""Subnetworks used in PIM."""

import numpy as np
from perceptual_quality.third_party.pyrtools import spfilters
import tensorflow.compat.v2 as tf


class Conv(tf.keras.Sequential):
  """Basic convolutional encoder."""

  def __init__(self, params, name='Conv'):
    super().__init__(name=name)
    num_filters = params['num_filters']
    for num in num_filters[:-1]:
      self.add(tf.keras.layers.Conv2D(
          num, 5, padding=params['padding'], activation=tf.nn.relu))
    self.add(tf.keras.layers.Conv2D(
        num_filters[-1], 5, padding=params['padding']))


class Steerable(tf.keras.layers.Layer):
  """Steerable pyramid."""

  def __init__(self, params, name='Steerable'):
    super().__init__(name=name)
    self.height = params['nscales']
    self.padding = params['padding'].upper()
    if self.padding not in ('VALID', 'SAME'):
      raise ValueError('Only valid or same padding is supported.')
    self.filter_type = params['steerable_filter_type']
    filters = getattr(spfilters, f'sp_filters{self.filter_type}')()
    self.nlevels = self.height * filters['bfilts'].shape[1] + 2

  def build(self, input_shape):
    super().build(input_shape)
    filters = getattr(spfilters, f'sp_filters{self.filter_type}')()

    self.lo0filt = tf.constant(
        filters['lo0filt'][:, :, None, None] * np.eye(input_shape[-1]),
        dtype=self.dtype)
    self.hi0filt = tf.constant(
        filters['hi0filt'][:, :, None, None] * np.eye(input_shape[-1]),
        dtype=self.dtype)
    self.lofilt = tf.constant(
        filters['lofilt'][:, :, None, None] * np.eye(input_shape[-1]),
        dtype=self.dtype)

    bfilters = filters['bfilts']
    nbands = bfilters.shape[1]
    bfilt_size = int(round(np.sqrt(bfilters.shape[0])))
    self.bfilts = []
    for b in range(nbands):
      filt = bfilters[:, b].reshape(bfilt_size, bfilt_size).T
      if (bfilt_size // 2) % 2:
        filt = np.pad(filt, [[1, 1], [1, 1]], mode='constant')
      filt = filt[:, :, None, None] * np.eye(input_shape[-1])
      self.bfilts.append(tf.constant(filt, dtype=self.dtype))

  def call(self, image):
    pyramid = []
    lo_padding = self.padding
    if self.padding == 'SAME':
      # Use explicit padding mode for image-size independent padding.
      lo_shape = [int(s) for s in self.lofilt.shape]
      assert lo_shape[0] % 2
      assert lo_shape[1] % 2
      lo_padding = [
          (0, 0),
          (lo_shape[0] // 2, lo_shape[0] // 2),
          (lo_shape[1] // 2, lo_shape[1] // 2),
          (0, 0),
      ]

    hi0 = tf.nn.conv2d(image, self.hi0filt, 1, padding=self.padding)
    pyramid.append(hi0)
    lo = tf.nn.conv2d(image, self.lo0filt, 1, padding=self.padding)

    for _ in range(self.height):
      for bfilt in self.bfilts:
        b = tf.nn.conv2d(lo, bfilt, 1, padding=self.padding)
        pyramid.append(b)
      lo = tf.nn.conv2d(lo, self.lofilt, 2, padding=lo_padding)
    pyramid.append(lo)

    return pyramid[::-1]


class Laplacian(tf.keras.layers.Layer):
  """Laplacian pyramid."""

  def __init__(self, params, name='Laplacian'):
    super().__init__(name=name)
    self.nlevels = params['nscales']
    self.use_residual = params['use_residual']

  def build(self, input_shape):
    super().build(input_shape)
    gauss_filter = (np.outer([1, 2, 1], [1, 2, 1])[:, :, None, None] / 16 *
                    np.eye(input_shape[-1]))
    self.gauss_filter = tf.constant(gauss_filter, dtype=self.dtype)

  def call(self, image):
    pyramid = []
    for i in range(self.nlevels):
      if i == self.nlevels-1 and self.use_residual:
        subband = image
      else:
        shape = tf.shape(image)
        low = tf.nn.conv2d(image, self.gauss_filter, 2, padding='VALID')
        image = image[:, 2:shape[1]-2, 2:shape[2]-2, :]
        high = tf.nn.conv2d_transpose(
            low, self.gauss_filter*4., tf.shape(image), 2,
            padding=((0, 0), (2, 2), (2, 2), (0, 0)))
        subband = image - high
        image = low
      pyramid.append(subband)
    return pyramid[::-1]
