# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
from mainmodels.models.tradition.config import g_MultiCNNConfig

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if g_MultiCNNConfig.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if g_MultiCNNConfig.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def create_conv_layer(tf_input, k_shape, k_stride, out_nums, layer_name,
                      stddev, wd=0.0):
    with tf.variable_scope(layer_name) as scope:
        chanel_input = tf_input.shape[-1].value
        k_shape = k_shape + [chanel_input, out_nums]
        kernel = _variable_with_weight_decay('weights',
                                             shape=k_shape,
                                             stddev=stddev,
                                             wd=wd)
        conv = tf.nn.conv2d(tf_input, kernel, k_stride, padding="SAME")
        biases = _variable_on_cpu('biases', [out_nums],
                                  tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv_layer = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv_layer)
    return conv_layer

def create_fc_layer(tf_input, out_nums, layer_name, stddev, wd):
    with tf.variable_scope(layer_name) as scope:
        input_dim_len = len(tf_input.shape)
        dim = 1
        for i in range(1, input_dim_len):
            dim *= tf_input.shape[i].value
        reshape = tf.reshape(tf_input, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, out_nums],
                                              stddev=stddev, wd=wd)
        biases = _variable_on_cpu('biases', [out_nums], tf.constant_initializer(0.1))
        fc_layer = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)
        _activation_summary(fc_layer)
    return fc_layer

def create_softmax_layer(tf_input, out_nums, layer_name, stddev=0.005, wd=0.0):
    with tf.variable_scope(layer_name) as scope:
        input_dim = tf_input.shape[-1].value
        weights = _variable_with_weight_decay('weights',
                                              [input_dim, out_nums],
                                              stddev=1/input_dim, wd=wd)
        biases = _variable_on_cpu('biases', [out_nums],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(tf_input, weights), biases,
                                name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear