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

from perceptual_quality.third_party.pyrtools import spfilters
import tensorflow as tf


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
    self.num_channels = params['num_channels']

  def build(self, input_shape):
    super().build(input_shape)
    filters = getattr(spfilters, f'sp_filters{self.filter_type}')()

    self.lo0filt = tf.tile(
        tf.constant(filters['lo0filt'], dtype=self.dtype)[:, :, None, None],
        [1, 1, self.num_channels, 1])
    self.hi0filt = tf.tile(
        tf.constant(filters['hi0filt'], dtype=self.dtype)[:, :, None, None],
        [1, 1, self.num_channels, 1])
    self.lofilt = tf.tile(
        tf.constant(filters['lofilt'], dtype=self.dtype)[:, :, None, None],
        [1, 1, self.num_channels, 1])

    bfilters = filters['bfilts']
    nbands = bfilters.shape[1]
    bfilt_size = int(round(bfilters.shape[0] ** .5))
    self.bfilts = []
    for b in range(nbands):
      filt = tf.constant(bfilters[:, b], dtype=self.dtype)
      filt = tf.transpose(tf.reshape(filt, (bfilt_size, bfilt_size)))
      if (bfilt_size // 2) % 2:
        filt = tf.pad(filt, [[1, 1], [1, 1]])
      self.bfilts.append(
          tf.tile(filt[:, :, None, None], [1, 1, self.num_channels, 1]))

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

    lo = tf.nn.depthwise_conv2d(
        image, self.lo0filt, strides=[1, 1, 1, 1], padding=self.padding)
    hi0 = tf.nn.depthwise_conv2d(
        image, self.hi0filt, strides=[1, 1, 1, 1], padding=self.padding)
    pyramid.append(hi0)

    for _ in range(self.height):
      for bfilt in self.bfilts:
        b = tf.nn.depthwise_conv2d(
            lo, bfilt, strides=[1, 1, 1, 1], padding=self.padding)
        pyramid.append(b)
      lo = tf.nn.depthwise_conv2d(
          lo, self.lofilt, strides=[1, 2, 2, 1], padding=lo_padding)
    pyramid.append(lo)

    return pyramid[::-1]


class Laplacian(tf.keras.layers.Layer):
  """Laplacian pyramid."""

  def __init__(self, params, name='Laplacian'):
    super().__init__(name=name)
    self.nlevels = params['nscales']
    self.use_residual = params['use_residual']
    self.num_channels = params['num_channels']

  def build(self, input_shape):
    super().build(input_shape)
    filt = tf.constant([1, 2, 1], dtype=self.dtype) / 4
    self.gauss_filter = tf.tile(
        filt[:, None, None, None] * filt[None, :, None, None],
        [1, 1, self.num_channels, 1])

  def call(self, image):
    pyramid = []
    transpose_filter = 4. * self.gauss_filter * tf.eye(self.num_channels)
    for i in range(self.nlevels):
      if i == self.nlevels-1 and self.use_residual:
        subband = image
      else:
        shape = tf.shape(image)
        low = tf.nn.depthwise_conv2d(
            image, self.gauss_filter, strides=[1, 2, 2, 1], padding='VALID')
        image = image[:, 2:shape[1]-2, 2:shape[2]-2, :]
        # There is no depthwise convolution implementation with upsampling, so
        # we need to use conv2d_transpose.
        high = tf.nn.conv2d_transpose(
            low, transpose_filter, tf.shape(image), 2,
            padding=((0, 0), (2, 2), (2, 2), (0, 0)))
        subband = image - high
        image = low
      pyramid.append(subband)
    return pyramid[::-1]
