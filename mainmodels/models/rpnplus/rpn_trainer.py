# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from mainmodels.models.rpnplus import data_engine
from mainmodels.models.rpnplus.config import g_RpnPlus_Config
from mainmodels.models.tradition.tools import extractTarget

image_height = g_RpnPlus_Config.image_height
image_width = g_RpnPlus_Config.image_width
feature_height = int(np.ceil(image_height / g_RpnPlus_Config.feature_ratio))
feature_width = int(np.ceil(image_width / g_RpnPlus_Config.feature_ratio))

class RPN:

    def build(self, image, label, label_weight, bbox_target, bbox_loss_weight,
              learning_rate):

        start_time = time.time()
        print('build model started')

        assert image.get_shape().as_list()[1:] == [image_height, image_width, 1]
        # Conv layer 1
        self.conv1_1, conv1_1_wd = self.conv_layer_new(image, 'conv1_1',
                                                       kernel_size=[5, 5],
                                                       out_channel=64)
        self.conv1_2, conv1_2_wd = self.conv_layer_new(self.conv1_1, 'conv1_2',
                                                       kernel_size=[5, 5],
                                                       out_channel=64)
        self.weight_dacay = conv1_1_wd + conv1_2_wd
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        # Conv layer 2
        self.conv2_1, conv2_1_wd = self.conv_layer_new(self.pool1, 'conv2_1',
                                                       kernel_size=[5, 5],
                                                       out_channel=128)
        self.conv2_2, conv2_2_wd = self.conv_layer_new(self.conv2_1, 'conv2_2',
                                                       kernel_size=[5, 5],
                                                       out_channel=128)
        self.weight_dacay += conv2_1_wd + conv2_2_wd
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Conv layer 3
        self.conv3_1, conv3_1_wd = self.conv_layer_new(self.pool2, 'conv3_1',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.conv3_2, conv3_2_wd = self.conv_layer_new(self.conv3_1, 'conv3_2',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.conv3_3, conv3_3_wd = self.conv_layer_new(self.conv3_2, 'conv3_3',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.weight_dacay += conv3_1_wd + conv3_2_wd + conv3_3_wd
        self.pool3 = self.max_pool(self.conv3_3, "pool3")
        # Conv lyer 4
        self.conv4_1, conv4_1_wd = self.conv_layer_new(self.pool3, 'conv4_1',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.conv4_2, conv4_2_wd = self.conv_layer_new(self.conv4_1, 'conv4_2',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.conv4_3, conv4_3_wd = self.conv_layer_new(self.conv4_2, 'conv4_3',
                                                       kernel_size=[3, 3],
                                                       out_channel=64)
        self.weight_dacay += conv4_1_wd + conv4_2_wd + conv4_3_wd

        normalization_factor = tf.sqrt(tf.reduce_mean(tf.square(self.conv4_3)))
        self.gamma2 = tf.Variable(np.sqrt(2), dtype=tf.float32, name='gamma3')
        self.gamma3 = tf.Variable(1.0, dtype=tf.float32, name='gamma4')

        # Pooling to the same size
        self.pool2_p = tf.nn.max_pool(self.pool2, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME',
                                      name='pool2_proposal')
        # L2 Normalization
        self.pool2_p = self.pool2_p / (
            tf.sqrt(tf.reduce_mean(
                tf.square(self.pool2_p))) / normalization_factor) * self.gamma2
        self.pool3_p = self.pool3 / (
            tf.sqrt(tf.reduce_mean(
                tf.square(self.pool3))) / normalization_factor) * self.gamma3

        # Proposal Convolution
        self.conv_proposal_2, conv_proposal_2_wd = self.conv_layer_new(
            self.pool2_p, 'conv_proposal_2',
            kernel_size=[5, 2], out_channel=128, stddev=0.01)
        self.relu_proposal_2 = tf.nn.relu(self.conv_proposal_2)
        self.weight_dacay += conv_proposal_2_wd

        self.conv_proposal_3, conv_proposal_3_wd = self.conv_layer_new(
            self.pool3_p, 'conv_proposal_3',
            kernel_size=[5, 2], out_channel=256, stddev=0.01)
        self.relu_proposal_3 = tf.nn.relu(self.conv_proposal_3)
        self.weight_dacay += conv_proposal_3_wd

        self.conv_proposal_4, conv_proposal_4_wd = self.conv_layer_new(
            self.conv4_3, 'conv_proposal_4',
            kernel_size=[5, 2], out_channel=256, stddev=0.01)
        self.relu_proposal_4 = tf.nn.relu(self.conv_proposal_4)
        self.weight_dacay += conv_proposal_4_wd

        # Concatrate
        self.relu_proposal_all = tf.concat([self.relu_proposal_2,
                                            self.relu_proposal_3,
                                            self.relu_proposal_4], 3)

        self.conv_cls_score, conv_cls_wd = self.conv_layer_new(
            self.relu_proposal_all, 'conv_cls_score',
            kernel_size=[1, 1], out_channel=18, stddev=0.01)
        self.conv_bbox_pred, conv_bbox_wd = self.conv_layer_new(
            self.relu_proposal_all, 'conv_bbox_pred',
            kernel_size=[1, 1], out_channel=36, stddev=0.01)
        self.weight_dacay += conv_cls_wd + conv_bbox_wd

        assert self.conv_cls_score.get_shape().as_list()[1:] == \
               [feature_height, feature_width, 18]
        assert self.conv_bbox_pred.get_shape().as_list()[1:] == \
               [feature_height, feature_width, 36]

        # shape=(batch_size*90*120*9, 2)
        self.cls_score = tf.reshape(self.conv_cls_score, [-1, 2])
        print("cls_score: ", self.cls_score.shape)
        # shape=(batch_size*90*120*9, 4)
        self.bbox_pred = tf.reshape(self.conv_bbox_pred, [-1, 4])
        print("bbox_pred: ", self.bbox_pred.shape)

        self.prob = tf.nn.softmax(self.cls_score, name="prob")
        self.cross_entropy = (tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.cls_score, labels=label) * label_weight) /
                              tf.reduce_sum(label_weight))

        bbox_error = tf.abs(self.bbox_pred - bbox_target)
        bbox_loss = (0.5 * bbox_error * bbox_error * tf.cast(
            bbox_error < 1, tf.float32) + (bbox_error - 0.5) * tf.cast(
            bbox_error >= 1, tf.float32))
        self.bb_loss = (tf.reduce_sum(
            tf.reduce_sum(bbox_loss, reduction_indices=[1]) *
            bbox_loss_weight) / tf.reduce_sum(bbox_loss_weight))

        self.loss = (self.cross_entropy + 0.0005 * self.weight_dacay + 0.5 *
                     self.bb_loss)

        self.train_step = tf.train.MomentumOptimizer(learning_rate,
                                                     0.9).minimize(self.loss)

        print('build model finished: %ds' % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer_new(self, bottom, name, kernel_size=None, out_channel=512,
                       stddev=0.01):
        if kernel_size is None:
            kernel_size = [3, 3]
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()[-1]
            filt = tf.Variable(
                tf.random_normal(
                    [kernel_size[0], kernel_size[1], shape, out_channel],
                    mean=0.0, stddev=stddev),
                name='filter')
            conv_biases = tf.Variable(tf.zeros([out_channel]), name='biases')

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            weight_dacay = tf.nn.l2_loss(filt, name='weight_dacay')
            return bias, weight_dacay

    def save(self, save_dir, step=None):
        params = {}
        for var in tf.trainable_variables():
            param_name = var.name.split('/')
            if param_name[1] in params.keys():
                params[param_name[1]].append(sess.run(var))
            else:
                params[param_name[1]] = [sess.run(var)]

        if step == None:
            step = 100000
        save_path = "/".join([save_dir, 'params_' + str(step) + '.npy'])
        np.save(save_path, params)


if __name__ == '__main__':
    gpuNow = '/cpu:0'
    print_time = 100
    step = 10000
    batch_size = 12
    saveTime = 2000

    modelSaveDir = g_RpnPlus_Config.save_rpnplus_model_dir
    train_samples_path = g_RpnPlus_Config.train_samples_index_path
    test_samples_path = g_RpnPlus_Config.test_samples_index_path
    if not os.path.exists(train_samples_path) or not os.path.exists(
            test_samples_path):
        extractTarget.prepare_rpn_list(train_samples_path, test_samples_path)

    with tf.device(gpuNow):
        sess = tf.Session()
        image = tf.placeholder(tf.float32, [None, image_height, image_width, 1])
        label = tf.placeholder(tf.float32, [None, 2])
        label_weight = tf.placeholder(tf.float32, [None])
        bbox_target = tf.placeholder(tf.float32, [None, 4])
        bbox_loss_weight = tf.placeholder(tf.float32, [None])
        learning_rate = tf.placeholder(tf.float32)

        cnn = RPN()
        with tf.name_scope('content_rpn'):
            cnn.build(image, label, label_weight, bbox_target, bbox_loss_weight,
                      learning_rate)

        sess.run(tf.global_variables_initializer())
        # for var in tf.trainable_variables():
        #     print(var.name, var.get_shape().as_list(),
        #           sess.run(tf.nn.l2_loss(var)))

        cnnData = data_engine.CNNData(train_samples_path, batch_size,
                                      original=False)
        print('Training Begin')

        train_loss = []
        train_cross_entropy = []
        train_bbox_loss = []
        start_time = time.time()

        for i in range(1, step + 1):
            batch = cnnData.prepare_data()
            if i <= 7000:
                l_r = 0.001
            else:
                if i <= 9000:
                    l_r = 0.0001
                else:
                    l_r = 0.00001
        #     (_, train_loss_iter, train_cross_entropy_iter, train_bbox_loss_iter,
        #      cls, bbox) = sess.run(
        #         [cnn.train_step, cnn.loss, cnn.cross_entropy, cnn.bb_loss,
        #          cnn.cls_score, cnn.bbox_pred],
        #         feed_dict={image: batch[0], label: batch[1],
        #                    label_weight: batch[2], bbox_target: batch[3],
        #                    bbox_loss_weight: batch[4], learning_rate: l_r})
        #
        #     train_loss.append(train_loss_iter)
        #
        #     if i % print_time == 0:
        #         print(' step :', i, 'time :', time.time() - start_time,
        #               'loss :', np.mean(
        #                 train_loss), 'l_r :', l_r)
        #         train_loss = []
        #
        #     if i % saveTime == 0:
        #         cnn.save(modelSaveDir, i)
