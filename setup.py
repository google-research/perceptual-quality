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
"""Setup script for perceptual-quality."""

import setuptools

with open("README.md", "r") as f:
  long_description = f.read()

with open("requirements.txt", "r") as f:
  install_requires = f.readlines()

setuptools.setup(
    name="perceptual-quality",
    version="0.1.dev",
    author="Google LLC",
    description="Perceptual quality metrics for TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/perceptual-quality",
    packages=setuptools.find_namespace_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
)
