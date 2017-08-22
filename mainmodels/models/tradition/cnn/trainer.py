from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import tensorflow as tf

# import __init
from mainmodels.models.tradition.cnn import model
from mainmodels.models.tradition.config import g_CNNConfig

absdirpath = os.path.dirname(os.path.abspath(__file__))

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', g_CNNConfig.train_log_dir,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 30000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = model.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images)

        # Calculate loss.
        loss = model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 50 == 0:
                    num_examples_per_step = g_CNNConfig.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = (
                    '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
