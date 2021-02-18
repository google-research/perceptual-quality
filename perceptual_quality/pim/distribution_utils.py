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
"""Helper functions for encoder distributions."""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Independent(object):
  """Joint distribution of independent multiscale components."""

  def __init__(self, components):
    self.components = components

  def log_prob(self, values):
    probs = [d.log_prob(v) for d, v in zip(self.components, values)]
    probs = [tf.reduce_mean(p, [-2, -1]) for p in probs]
    probs = tf.reduce_mean(probs, 0)  # return is [s,b]
    return probs

  def sample(self, sample_shape=()):
    return [distb.sample(sample_shape) for distb in self.components]

  def loc(self):
    return [distb.loc for distb in self.components]

  def mean(self):
    return [distb.mean() for distb in self.components]


class EncoderDist(tf.keras.layers.Layer):
  """Common encoder distribution class."""

  def __init__(self, nlayers, nz, nmix=1, joint=False, name=None):
    """Build requested encoder distribution object.

    Args:
      nlayers: Hyperparameter for number of layers in each network block.
      nz: Last dimension of z (depth at each spatial location in z).
      nmix: Number of mixture components in marginal encoders (=1 for joint
        encoder).
      joint: A boolean variable, set to True for returning joint encoder
        distributions instead of marginal.
      name: Name for distribution object.

    Returns:
      Joint encoder distribution object if joint is True and marginal otherwise.
    """
    super().__init__(name=name)
    self.nmix = max(1, nmix)
    self.nz = nz
    self.nlayers = nlayers
    self.joint = joint

    self.base = [
        tf.keras.layers.Dense(nz * nmix, activation=tf.nn.relu)
        for _ in range(nlayers)
    ]
    self.mus_layer = tf.keras.layers.Dense(nz * nmix)

    if self.joint:
      self.base_cond_mul = [
          tf.keras.layers.Dense(nz * nmix) for _ in range(nlayers)
      ]
      self.base_cond_add = [
          tf.keras.layers.Dense(nz * nmix) for _ in range(nlayers)
      ]

    if nmix > 1:
      self.mix = [
          tf.keras.layers.Dense(nmix, activation=tf.nn.relu)
          for _ in range(nlayers)
      ]
      self.logit_layer = tf.keras.layers.Dense(nmix)

  def call(self, f_x, f_y=None):
    """Returns encoder distribution object.

    Args:
      f_x: Output of frontend for an image x.
      f_y: Output of frontend for second image, should only be passed if
        joint encoder distribution is required (self.joint is True).

    Returns:
      Joint encoder distribution object if self.joint is True and marginal
      encoder distribution otherwise.
    """
    if self.joint and f_y is None:
      raise ValueError('f_y is None (joint=True)')
    if not self.joint and f_y is not None:
      raise ValueError('f_y is not None (joint=False)')

    if self.nmix > 1:
      mix_net = f_x

    net = f_x
    for i in range(self.nlayers):
      net = self.base[i](net)
      if self.joint:
        net *= self.base_cond_mul[i](f_y)
        net += self.base_cond_add[i](f_y)
      if self.nmix > 1:
        mix_net = self.mix[i](mix_net)
    if net.shape.as_list()[-1] == self.nz * self.nmix:
      mus = net
    else:
      mus = self.mus_layer(net)
    if self.nmix > 1:
      mus = tf.reshape(
          mus, tf.concat((tf.shape(mus)[:-1], [self.nmix, self.nz]), axis=0))

    dist = tfd.MultivariateNormalDiag(loc=mus)
    if self.nmix > 1:
      logits = self.logit_layer(mix_net)
      mix_dist = tfd.Categorical(logits=logits)
      dist = tfd.MixtureSameFamily(mixture_distribution=mix_dist,
                                   components_distribution=dist)
    return dist


class MarginalEncoder(tf.keras.layers.Layer):
  """Multiscale marginal encoder distribution q(z|x)."""

  def __init__(self, nlevels, nlayers, nz, nmix):
    super().__init__(name='marginal_encoder')
    self.encoders = [EncoderDist(nlayers, nz, nmix) for _ in range(nlevels)]

  def call(self, f_xs):
    dists = []

    if len(f_xs) != len(self.encoders):
      raise ValueError('Number of levels in marginal encoder (%d) not equal to '
                       'number of levels in the frontend\'s output (%d).'
                       % (len(self.encoders), len(f_xs)))
    for f_x, encoder in zip(f_xs, self.encoders):
      dists.append(encoder(f_x))
    return Independent(dists)


class JointEncoder(tf.keras.layers.Layer):
  """Multiscale joint encoder distribution p(z|x,y)."""

  def __init__(self, nlevels, nlayers, nz):
    super().__init__(name='joint_encoder')
    self.encoders = [
        EncoderDist(nlayers, nz, joint=True) for _ in range(nlevels)
    ]

  def call(self, f_xs, f_ys):
    dists = []
    if not len(f_xs) == len(f_ys) == len(self.encoders):
      raise ValueError('Number of levels in joint encoder (%d) not equal to '
                       'number of levels in the frontend\'s output (%d, %d).'
                       % (len(self.encoders), len(f_xs), len(f_ys)))
    for f_x, f_y, encoder in zip(f_xs, f_ys, self.encoders):
      dists.append(encoder(f_x, f_y=f_y))
    return Independent(dists)
