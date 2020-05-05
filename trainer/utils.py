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

"""Common functions for building TFGAN models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

import tensorflow as tf
from tensorflow.contrib.gan.python.eval import summaries as gansummaries
from tensorflow.contrib.gan.python.losses.python import losses_impl as ganloss

from constants import constants
from trainer import metrics


def summary_image_normalized(label, image):
  """Returns normalized image summary TF op."""
  return tf.summary.image(label, tf.image.convert_image_dtype(image, tf.uint8,
                                                              saturate=True))


def get_profiler_hook(job_dir):
  """Returns tf.train.ProfilerHook."""
  return tf.train.ProfilerHook(
      save_steps=constants.PROFILING_SAVE_STEPS,
      output_dir=os.path.join(job_dir, constants.PROFILING_SUB_DIR))


def batchnorm(inputs, mode=tf.estimator.ModeKeys.TRAIN):
  """Computes batch normalization.

  Args:
    inputs: Input tensor.
    mode: tf.estimator.ModeKeys mode.

  Returns:
    Batch norm output tensor.
  """
  training = mode not in (tf.estimator.ModeKeys.EVAL,
                          tf.estimator.ModeKeys.PREDICT)
  outputs = tf.layers.batch_normalization(inputs, training=training)
  return outputs


def conv2d(inputs, filters=64, kernel_width=3, stride_width=1):
  """Convolution function wrapper to consolidate with default args."""
  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=(kernel_width, kernel_width),
      strides=(stride_width, stride_width),
      padding='same')


def residual_block(
    inputs,
    filters=64,
    kernel_width=3,
    stride_width=1,
    mode=tf.estimator.ModeKeys.TRAIN):
  """Residual Network sub-blocks."""
  block_inputs = inputs
  inputs = conv2d(
      inputs,
      filters=filters,
      kernel_width=kernel_width,
      stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d(
      inputs,
      filters=filters,
      kernel_width=kernel_width,
      stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs += block_inputs
  return inputs


def conv_block(inputs, filters, stride_width, mode=tf.estimator.ModeKeys.TRAIN):
  """Discriminator convolutional sub-blocks."""
  inputs = conv2d(inputs, filters=filters, stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs = tf.nn.leaky_relu(inputs)
  return inputs


def conv3d(inputs, filters=64, kernel_width=3, stride_width=1):
  """Convolution function wrapper for 3D to consolidate with default args."""
  return tf.layers.conv3d(
      inputs=inputs,
      filters=filters,
      kernel_size=(kernel_width, kernel_width, kernel_width),
      strides=(stride_width, stride_width, stride_width),
      padding='same')


def residual3d_block(
    inputs,
    filters=64,
    kernel_width=3,
    stride_width=1,
    mode=tf.estimator.ModeKeys.TRAIN):
  """Residual Network sub-blocks for 3D."""
  block_inputs = inputs
  inputs = conv3d(
      inputs,
      filters=filters,
      kernel_width=kernel_width,
      stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs = tf.nn.relu(inputs)
  inputs = conv3d(
      inputs,
      filters=filters,
      kernel_width=kernel_width,
      stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs += block_inputs
  return inputs


def conv3d_block(
    inputs,
    filters=64,
    stride_width=1,
    mode=tf.estimator.ModeKeys.TRAIN):
  """Discriminator convolutional sub-blocks for 3D."""
  inputs = conv3d(inputs, filters=filters, stride_width=stride_width)
  inputs = batchnorm(inputs, mode)
  inputs = tf.nn.leaky_relu(inputs)
  return inputs


## Loss functions ##
def combined_generator_loss(
    gan_model,
    perceptual_loss_order=2,
    weight_factor=1.0e-03,
    gradient_ratio=None,
    num_comparisons=10,
    add_summaries=True):
  """Adversarial loss and L1 perceptual loss.
  Loss functions are one of the only points that the GANModel is exposed,
  it is used here for additional summaries and evaluation metrics.

  Args:
    gan_model: TFGAN named tuple for handling gen/disc inputs/outputs.
    perceptual_loss_order: 1 for L1 loss, 2 for L2 loss in gen/hr pixel diff.
    weight_factor: adversarial loss coeff, None if using gradient_ratio
    gradient_ratio: relative magnitude of perceptual and adversarial loss,
        None if using weight factor
    num_comparisons: The number of lr/gen/hr image comparisons to display per
        summary. Defaults to 10, the maximum number of 128x128 images that fits
        widthwise on a 720p display.
    add_summaries: Whether to add loss summaries, expected by GANEstimator.

  Returns:
    generator_loss_fn combining the adversarial and L1 perceptual losses.
  """
  discriminator_gen_outputs = gan_model.discriminator_gen_outputs
  hr_image = gan_model.real_data
  gen_image = gan_model.generated_data['gen_image']
  lr_image = gan_model.generator_inputs['lr_image']

  adversarial_loss = ganloss.modified_generator_loss(
      discriminator_gen_outputs=discriminator_gen_outputs,
      add_summaries=add_summaries)

  perceptual_loss = tf.norm(
      gen_image - hr_image,
      ord=perceptual_loss_order)

  combined_adversarial_loss = ganloss.combine_adversarial_loss(
      perceptual_loss,
      adversarial_loss,
      weight_factor=weight_factor,
      gradient_ratio=gradient_ratio)

  add_scalar_summaries(hr_image, gen_image)
  add_image_comparison_summaries(hr_image, gen_image, lr_image, num_comparisons)

  return combined_adversarial_loss


def add_scalar_summaries(hr_image, gen_image):
  """Adds scalar summaries to measure accuracy of predictions in training.

  Args:
    hr_image: tensor of high resolution images
    gen_image: tensor of generated images
  """
  tf.summary.scalar('SSIM', metrics.ssim(hr_image, gen_image))
  tf.summary.scalar('PSNR', metrics.psnr(hr_image, gen_image))
  tf.summary.scalar('MSE', tf.losses.mean_squared_error(hr_image, gen_image))


def depth_to_batch(image):
  """If images are 3D, convert to batches of depthwise 2D image slices.
  Args:
    image: (batch, height, width, depth, channels)

  Returns:
    images with (batch * depth, height, width, channels)

  If the input is already a 2d image, return it unaltered.
  """
  if len(image.shape) == 4:
    return image

  batch, height, width, depth, channels = image.shape.as_list()
  return tf.reshape(image, (batch * depth, height, width, channels))


def shuffle_batch(tensors):
  """Repeats the same shuffle for a list of tensors along the batch dimension.

  Args:
    tensors: a list of of tensors with batch as the first dimension.

  Returns:
    a list of shuffled tensors
  """
  tensors = tf.stack(tensors, axis=-1)
  tensors = tf.random_shuffle(tensors)
  return tf.unstack(tensors, axis=-1)


def add_image_comparison_summaries(
    hr_image, gen_image, lr_image, num_comparisons):
  """Builds a gan_model namedtuple and calls tfgans image comparison summary.

  Args:
    hr_image: tensor of high resolution images
    gen_image: tensor of generated images
    lr_image: tensor of low resolution images
    num_comparisons: number of image comparisons to save per summary

  For 3d images, xz slices are made and the y dimension is stacked in the batch
  dimension. The batch is then shuffled so the comparison isn't a single block.

  in the event that num_comparisons is larger than the resulting batch size, the
  batch size is used instead.
  """

  hr_image = depth_to_batch(hr_image)
  gen_image = depth_to_batch(gen_image)
  lr_image = depth_to_batch(lr_image)

  hr_image, gen_image, lr_image = shuffle_batch([hr_image, gen_image, lr_image])

  hr_image = tf.map_fn(tf.image.per_image_standardization, hr_image)
  gen_image = tf.map_fn(tf.image.per_image_standardization, gen_image)
  lr_image = tf.map_fn(tf.image.per_image_standardization, lr_image)

  # Field names are taken to be consistent with gan_model from TFGAN.
  summary_gan_model = collections.namedtuple(
      'generated_data', 'real_data', 'generator_inputs')
  summary_gan_model.real_data = hr_image
  summary_gan_model.generated_data = gen_image
  summary_gan_model.generator_inputs = lr_image

  batch_size = lr_image.shape.as_list()[0]

  gansummaries.add_image_comparison_summaries(
      gan_model=summary_gan_model,
      num_comparisons=min(num_comparisons, batch_size))


# Evaluation metrics
def get_eval_metric_ops_fn(gan_model):
  """Function to handle evaluation metrics needed by GANEstimator.

  Args:
    gan_model: tf.contrib.GANModel namedtuple

  Returns:
    eval_metric_ops: dict of MetricSpec

  The prefix of GANHead is added so the naming matches training summaries and
  they share the same plot in TensorBoard.
  """
  hr_image = gan_model.real_data
  gen_image = gan_model.generated_data['gen_image']
  eval_metric_ops = {
      'GANHead/MSE': tf.metrics.mean_squared_error(hr_image, gen_image),
      'GANHead/SSIM': tf.metrics.mean(metrics.ssim(hr_image, gen_image)),
      'GANHead/PSNR': tf.metrics.mean(metrics.psnr(hr_image, gen_image)),
  }
  return eval_metric_ops
