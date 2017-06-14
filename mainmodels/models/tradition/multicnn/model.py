# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from mainmodels.models.tradition import data_input
from mainmodels.models.tradition.config import g_MultiCNNConfig
from mainmodels.models.tradition.multicnn.layers import _add_loss_summaries
from mainmodels.models.tradition.multicnn.layers import create_conv_layer
from mainmodels.models.tradition.multicnn.layers import create_fc_layer
from mainmodels.models.tradition.multicnn.layers import create_softmax_layer


def distorted_inputs():
    images, labels = data_input.distorted_inputs(
        filename=g_MultiCNNConfig.train_samplesp_path,
        batch_size=g_MultiCNNConfig.batch_size)
    if g_MultiCNNConfig.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inputs(test_sample_path):
    images, labels = data_input.inputs(test_sample_path=test_sample_path,
                                       batch_size=g_MultiCNNConfig.batch_size)
    if g_MultiCNNConfig.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def cnn_model_01(images):
    conv1 = create_conv_layer(images, k_shape=[5, 5], k_stride=[1, 2, 2, 1],
                              out_nums=6, layer_name="conv1", stddev=5e-2,
                              wd=0.0)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool1")
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    conv2 = create_conv_layer(norm1, k_shape=[7, 7], k_stride=[1, 1, 1, 1],
                              out_nums=12, layer_name="conv2", stddev=5e-2,
                              wd=0.0)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool2")

    return pool2

def cnn_model_02(images):
    conv1 = create_conv_layer(images, k_shape=[7, 7], k_stride=[1, 2, 2, 1],
                              out_nums=6, layer_name="conv1", stddev=5e-2,
                              wd=0.0)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool1")
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    conv2 = create_conv_layer(norm1, k_shape=[5, 5], k_stride=[1, 1, 1, 1],
                              out_nums=12, layer_name="conv2", stddev=5e-2,
                              wd=0.0)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool2")

    return pool2

def cnn_model_03(images):
    conv1 = create_conv_layer(images, k_shape=[3, 3], k_stride=[1, 1, 1, 1],
                              out_nums=12, layer_name="conv1", stddev=5e-2,
                              wd=0.0)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool1")
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    conv2 = create_conv_layer(norm1, k_shape=[3, 3], k_stride=[1, 1, 1, 1],
                              out_nums=12, layer_name="conv2", stddev=5e-2,
                              wd=0.0)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool2")
    conv3 = create_conv_layer(pool2, k_shape=[3, 3], k_stride=[1, 1, 1, 1],
                              out_nums=24, layer_name="conv3", stddev=5e-2,
                              wd=0.0)
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding="SAME", name="pool3")

    return pool3

def inference(images):
    with tf.variable_scope("cnn_model_01"):
        cnn_01_output = cnn_model_01(images)
    with tf.variable_scope("cnn_model_02"):
        cnn_02_output = cnn_model_02(images)
    with tf.variable_scope("cnn_model_03"):
        cnn_03_output = cnn_model_03(images)

    pool_all = tf.concat([cnn_01_output,
                          cnn_02_output,
                          cnn_03_output], 3)

    local3 = create_fc_layer(pool_all, out_nums=24, layer_name="local3",
                             stddev=0.04, wd=0.004)
    softmax_linear = create_softmax_layer(local3, g_MultiCNNConfig.NUM_CLASSES,
                                          "softmax_linear", stddev=5e-2, wd=0.0)
    return softmax_linear

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
    # Variables that affect learning rate.
    num_batches_per_epoch = g_MultiCNNConfig.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /\
                            g_MultiCNNConfig.batch_size
    decay_steps = int(num_batches_per_epoch * g_MultiCNNConfig.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(g_MultiCNNConfig.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    g_MultiCNNConfig.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        g_MultiCNNConfig.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

if __name__ == '__main__':
    images = tf.placeholder(dtype=tf.float32, shape=(None, 48, 48, 1))
    inference(images)
