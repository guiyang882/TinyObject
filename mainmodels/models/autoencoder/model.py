# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/5/22

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

class SingleLayerAE():

    def _pad(self, input_x, filter_side):
        amount = filter_side - 1
        paddings = [[0, 0], [amount, amount], [amount, amount], [0, 0]]
        return tf.pad(input_x, paddings=paddings)

    def get(self, images, train_phase=False, l2_penalty=0.0):
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=2.0,
            mode='FAN_IN',
            uniform=False,
            seed=69,
            dtype=tf.float32)
        filter_side = 3
        filters_number = 32

        with tf.variable_scope("prepare"):
            input_x = self._pad(images, filter_side)

        with tf.variable_scope("encode"):
            encoding = conv(
                input_x=input_x,
                shape=[filter_side, filter_side, input_x.get_shape()[3].value, filters_number],
                stride=1,
                padding="VALID",
                train_phase=train_phase,
                activation=tf.nn.tanh,
                wd=l2_penalty,
                initializer=initializer
            )
        # encode layer info (32, 34, 34, 32)
        print("encode layer info", encoding.get_shape())
        with tf.variable_scope("decode"):
            decoding = conv(
                input_x=encoding,
                shape=[filter_side, filter_side, filters_number, input_x.get_shape()[3].value],
                stride=1,
                padding="VALID",
                train_phase=train_phase,
                activation=tf.nn.tanh,
                initializer=initializer
            )

        is_training_ = tf.placeholder_with_default(
            False, shape=(), name="is_training_"
        )
        return is_training_, decoding

    def loss(self, predictions, real_values):
        with tf.variable_scope("loss"):
            mse = tf.divide(
                tf.reduce_mean(
                    tf.square(tf.subtract(predictions, real_values))
                ), 2.0, name="mse")
        return mse