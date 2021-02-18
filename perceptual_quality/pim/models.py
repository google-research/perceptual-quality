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
"""Models used in metric."""

from perceptual_quality.pim import distribution_utils
from perceptual_quality.pim import networks
import tensorflow.compat.v2 as tf


class SingleScaleFrontend(tf.keras.layers.Layer):
  """Single scale frontend."""

  def __init__(self, params):
    super().__init__()
    self.cnn = getattr(networks, params['cnn'])(params)
    self.nlevels = 1

  def call(self, image):
    return [self.cnn(image)]


class MultiScaleFrontend(tf.keras.layers.Layer):
  """Multiple scale frontend."""

  def __init__(self, params):
    super().__init__()
    self.multiscale = getattr(networks, params['multiscale'])(params)
    self.nlevels = self.multiscale.nlevels

    cnn_arch = getattr(networks, params['cnn'])
    self.cnns = []
    for _ in range(self.nlevels):
      self.cnns.append(cnn_arch(params))

  def call(self, image):
    pyramid = self.multiscale(image)
    assert len(self.cnns) == len(pyramid)
    for i, cnn in enumerate(self.cnns):
      pyramid[i] = cnn(pyramid[i])

    return pyramid


class PIM(tf.keras.Model):
  """Perceptual Information Metric."""

  def __init__(self, params):
    super().__init__()

    if params['multiscale'] is not None:
      self.frontend = MultiScaleFrontend(params)
    else:
      self.frontend = SingleScaleFrontend(params)

    self.nmix = params['num_marginal_encoder_mixtures']
    self.marginal_encoder = distribution_utils.MarginalEncoder(
        self.frontend.nlevels,
        nlayers=params['num_distribution_encoder_layers'],
        nz=params['z_dim'],
        nmix=params['num_marginal_encoder_mixtures'])
    self.joint_encoder = distribution_utils.JointEncoder(
        self.frontend.nlevels,
        nlayers=params['num_distribution_encoder_layers'],
        nz=params['z_dim'])

  def call(self, inputs, return_distributions=False, kld_samples=1):
    """Calculates distance between images.

    The inputs are assumed to be floating-point sRGB images, with a dynamic
    range between 0 and 1. Image `Tensor`s are expected to have NHWC format
    (with a leading batch dimension and a trailing channel dimension of length
    3).

    Args:
      inputs: A tuple of 2 images `(x, y)` or 3 images `(x, y, y2)`. A 3-tuple
        corresponds to the triplet setting, where `x` is the reference image,
        and `y` and `y2` are distorted images.
      return_distributions: Boolean. Set to `True` to return encoder
        distributions instead of distance. The value is ignored and assumed to
        be `True` when `training==True`.
      kld_samples: Integer. Number of samples to use for estimating symmetrized
        Kullback–Leibler divergence. Ignored if the marginal encoder
        distributions have only one mixture component (since the KLD then
        collapses to the squared Euclidean distance).

    Returns:
      If `return_distributions` or `training` is `True`, returns encoder
      distributions. Otherwise, returns the distance between `x` and `y`. If
      `y2` is provided, also returns the distance between `x` and `y2`.
    """
    if len(inputs) == 2:
      x, y = inputs
      y2 = None
    elif len(inputs) == 3:
      x, y, y2 = inputs
    else:
      raise ValueError('%d images passed to PIM, only 2 or 3 allowed.' %
                       len(inputs))

    f_x = self.frontend(x)
    f_y = self.frontend(y)
    f_y2 = self.frontend(y2) if y2 is not None else None

    zx_dist = self.marginal_encoder(f_x)
    zy_dist = self.marginal_encoder(f_y)
    if y2 is not None:
      zy2_dist = self.marginal_encoder(f_y2)

    if return_distributions:
      distributions = {
          'z_x': zx_dist,
          'z_y': zy_dist,
      }
      if y2 is not None:
        distributions.update({'z_y2': zy2_dist})
      return distributions

    distance = self._get_distance(zx_dist, zy_dist, kld_samples)
    if y2 is not None:
      distance2 = self._get_distance(zx_dist, zy2_dist, kld_samples)
      return distance, distance2
    return distance

  def _get_distance(self, dist_x, dist_y, num_samples):
    if self.nmix == 1:
      mean_x = dist_x.mean()
      mean_y = dist_y.mean()

      distances = []
      assert len(mean_x) == len(mean_y)
      for zx, zy in zip(mean_x, mean_y):
        diff = tf.math.squared_difference(zx, zy)
        distances.append(tf.reduce_mean(diff, range(1, diff.shape.rank)))
      return tf.reduce_mean(distances, axis=0)
    else:
      def kl(dist1, dist2):
        x_samples = dist1.sample(num_samples)
        log_ratio = dist1.log_prob(x_samples) - dist2.log_prob(x_samples)
        return tf.reduce_mean(log_ratio, axis=0)

      return kl(dist_x, dist_y) + kl(dist_y, dist_x)
