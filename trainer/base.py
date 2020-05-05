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

"""Base ML model definitions."""

import abc


import functools
import tensorflow as tf
import tensorflow.contrib.gan as tfgan

from trainer.model import utils

class AbstractStaticMethod(staticmethod):
  """Implement decorator enforcing abstract and static schemes.

  This is a workaround.
  Versions below Python 3.2+ do not yet support abc.AbstractStaticMethod
  The following is a workaround.
  Reference:
  (https://stackoverflow.com/questions/4474395/
   staticmethod-and-abc-abstractmethod-will-it-blend?)
  """
  __slots__ = ()

  def __init__(self, function):
    super(AbstractStaticMethod, self).__init__(function)
    function.__isabstractmethod__ = True

  __isabstractmethod__ = True


class BaseModel(object):
  """Base class for ML models."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, hparams, run_config):
    """Store hparams and run_config as attributes."""
    self._hparams = hparams
    self._run_config = run_config

  @property
  def hparams(self):
    """Returns tf.contrib.training.HParams object."""
    return self._hparams

  @property
  def run_config(self):
    """Returns tf.estimator.RunConfig."""
    return self._run_config

  @abc.abstractmethod
  def build_estimator(self):
    """Should return a tf.estimator.Estimator object."""
    return

  def train_and_evaluate(self, train_input_fn, eval_input_fn,
                         serving_input_receiver_fn):
    """Convenience wrapper that calls build_estimator().train_and_evaluate.

    Args:
      train_input_fn: Training input function.
      eval_input_fn: Evaluation input function.
      serving_input_fn_map: Dict of {'input_type': serving_input_fn()}.
    """

    estimator = self.build_estimator()

    # Setup training hooks
    train_hooks = []
    if self.hparams.profiling:
      train_hooks.append(
          utils.get_profiler_hook(self.hparams.job_dir))

    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=self.hparams.train_steps, hooks=train_hooks)

    final_exporter = tf.estimator.FinalExporter(
        'final_exporter', serving_input_receiver_fn)

    # TODO(andrewleach): reset throttle default once multigpu no longer hangs.
    # It is a temporary solution to enforce final eval and export as every
    # checkpoint gets evaluated, and global_steps > max_steps triggers export.
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=self.hparams.eval_steps,
        exporters=[final_exporter])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


class BaseGANModel(BaseModel):
  """Base class for GAN models.

  Wraps tf.contrib.gan.GANEstimator and converts keras models returned
  by generator_fn() and discriminator_fn() to appropriate TF ops
  needed by GANEstimator.
  """

  @abc.abstractmethod
  def get_input_shape(self):
    """Returns input shape as a tuple."""

    raise NotImplementedError

# pylint: disable=unused-argument

  @AbstractStaticMethod
  def generator_model(input_shape, hparams):
    """Keras model representing generator network.

    Implement as @staticmethod in subclass.

    Args:
      input_shape: Input shape as tuple.

    Returns:
      Should return a model which accepts inputs with input_shape and the mode
      as tf.estimator.ModeKeys.
    """
    raise NotImplementedError

  @AbstractStaticMethod
  def discriminator_model(input_shape, hparams):
    """Keras model representing discriminator network.

    Implement as @staticmethod in subclass.

    Args:
      input_shape: Input shape as tuple.

    Returns:
      Should return a model which accepts inputs with input_shape and the mode
      as tf.estimator.ModeKeys.
    """
    raise NotImplementedError

  def _make_generator_fn(self):
    """Returns generator_fn() to pass to TF GANEstimator."""

    def generator_fn(
        inputs, mode=tf.estimator.ModeKeys.TRAIN):
      """Returns output of generator network.

      Args:
        inputs: Inputs to the generator, will always be
            {'lr_image':[...], 'key':[...]}
        mode: Mode keys (tf.estimator.ModeKeys)
      """

      gen_image = self.generator_model(
          self.get_input_shape(), self.hparams)(inputs['lr_image'], mode)
      return {'gen_image': gen_image, 'key': inputs['key']}

    return generator_fn

  def _make_discriminator_fn(self):
    """Returns discriminator_fn() to pass to TF GANEstimator."""

    def discriminator_fn(
        inputs, conditioning=None, mode=tf.estimator.ModeKeys.TRAIN):
      """Returns output of discriminator network.

      Args:
        inputs: Real/generator input, will be one of the following
            generated_data := {'gen_image':[...], 'key':[...]}
            real_data := [...]
        conditioning: Additional conditioning information.
        mode: Mode keys (tf.estimator.ModeKeys)
      """

      # generated_data will be a dict, real_data will be a tensor.
      if isinstance(inputs, dict):
        inputs = inputs['gen_image']
      return self.discriminator_model(
          self.get_input_shape(), self.hparams)(inputs, mode)

    return discriminator_fn

  @staticmethod
  def get_generator_loss_fn(weight_factor=1.0e-03):
    """Returns generator loss."""
    return functools.partial(
        utils.combined_generator_loss,
        weight_factor=weight_factor)

  @staticmethod
  def get_discriminator_loss_fn():
    """Returns discriminator loss."""

    return tfgan.losses.modified_discriminator_loss

  def generator_optimizer(self):
    """Returns generator optimizer."""

    return tf.train.AdamOptimizer(self.hparams.gen_learning_rate)

  def discriminator_optimizer(self):
    """Returns discriminator optimizer."""

    return tf.train.AdamOptimizer(self.hparams.disc_learning_rate)

  def build_estimator(self):

    return tfgan.estimator.GANEstimator(
        generator_fn=self._make_generator_fn(),
        discriminator_fn=self._make_discriminator_fn(),
        generator_loss_fn=self.get_generator_loss_fn(
            weight_factor=self._hparams.weight_factor),
        discriminator_loss_fn=self.get_discriminator_loss_fn(),
        generator_optimizer=self.generator_optimizer(),
        discriminator_optimizer=self.discriminator_optimizer(),
        get_eval_metric_ops_fn=utils.get_eval_metric_ops_fn,
        config=self.run_config)
