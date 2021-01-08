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
"""Base parameters."""


def get_params(**override):
  """Returns default parameters dictionary for model."""

  params = dict(
      # Model args
      num_channels=3,
      multiscale='Steerable',  # 'Steerable', Laplacian', None
      nscales=3,
      steerable_filter_type=1,

      cnn='Conv',
      num_filters=[64, 64, 64, 3],
      padding='same',

      # Distribution args
      num_distribution_encoder_layers=2,
      num_marginal_encoder_mixtures=5,
      z_dim=10,
  )
  params.update(override)
  return params
