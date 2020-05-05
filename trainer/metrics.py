# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Contains the Metric functions for evaluating our model."""

from __future__ import print_function

import functools

from skimage import measure
import numpy as np
import tensorflow as tf


def ssim(hr_image, gen_image):
  """Calculates the average SSIM value over a batch.

  Args:
    hr_image: batch of 2D/3D images or single image
    gen_image: batch of 2D/3D images or single image

  Returns:
    average SSIM value
  """
  def _ssim(hr_image, gen_image):
    """python implementation of SSIM averaged over batch dimension."""
    single_ssim = functools.partial(
        measure.compare_ssim,
        gaussian_weights=True,
        multichannel=True,
        data_range=1.0)
    return np.mean(map(single_ssim, hr_image, gen_image), dtype='float32')
  return tf.py_func(_ssim, [hr_image, gen_image], tf.float32)


def psnr(hr_image, gen_image):
  """Calculates the average PSNR value over a batch.

  Args:
    hr_image: batch of 2D/3D images or single image
    gen_image: batch of 2D/3D images or single image

  Returns:
    average PSNR value
  """
  def _psnr(hr_image, gen_image):
    """python implementation of SSIM averaged over batch dimension."""
    single_psnr = functools.partial(measure.compare_psnr, data_range=1)
    return np.mean(map(single_psnr, hr_image, gen_image), dtype='float32')
  return tf.py_func(_psnr, [hr_image, gen_image], tf.float32)
