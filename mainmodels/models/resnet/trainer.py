# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mainmodels.dataset import cifar_input
from mainmodels.models.resnet import model as resnet_model
from mainmodels.models.resnet.settings import g_ResNetConfig


def train(hps):
    images, labels = cifar_input.build_input(
        g_ResNetConfig.dataset, g_ResNetConfig.train_data_path,
        hps.batch_size, g_ResNetConfig.mode)
    model = resnet_model.ResNet(hps, images, labels, g_ResNetConfig.mode)
    model.build_graph()

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=g_ResNetConfig.train_dir,
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=10)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.1

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 40000:
                self._lrn_rate = 0.1
            elif train_step < 60000:
                self._lrn_rate = 0.01
            elif train_step < 80000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0001

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=g_ResNetConfig.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)

def main(_):
    if g_ResNetConfig.num_gpus == 0:
        dev = '/cpu:0'
    elif g_ResNetConfig.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if g_ResNetConfig.mode == 'train':
        batch_size = 32
    elif g_ResNetConfig.mode == 'eval':
        batch_size = 100

    if g_ResNetConfig.dataset == 'cifar10':
        num_classes = 10
    elif g_ResNetConfig.dataset == 'cifar100':
        num_classes = 100

    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=2,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')

    with tf.device(dev):
        if g_ResNetConfig.mode == 'train':
            train(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()