# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/5/22

from __future__ import absolute_import
from __future__ import print_function

import os

import tensorflow as tf
from mainmodels.models.autoencoder.builders import build_batch, scale_image

from mainmodels.models.autoencoder.utils import InputType

filepath = "../../output/0001_252004_32_32.bin"

if not os.path.exists(filepath):
    raise IOError(filepath + " not found !")

class WordData():

    def __init__(self):
        self._name = "WordData"
        self._num_examples_per_epoch_for_train = 180780
        self._num_examples_per_epoch_for_eval = 45196
        self._num_examples_per_epoch_for_test = 45196
        self._num_classes = 10

    def num_examples(self, input_type):
        """Returns the number of examples per the specified input_type

        Args:
            input_type: InputType enum
        """
        InputType.check(input_type)

        if input_type == InputType.train:
            return self._num_examples_per_epoch_for_train
        elif input_type == InputType.test:
            return self._num_examples_per_epoch_for_test
        return self._num_examples_per_epoch_for_eval

    def inputs(self, input_type, batch_size):
        # InputType.check(input_type=input_type)
        if input_type == InputType.train:
            num_examples_per_epoch = self._num_examples_per_epoch_for_train
            filenames = [filepath]
        elif input_type == InputType.test:
            num_examples_per_epoch = self._num_examples_per_epoch_for_test
            filenames = [filepath]
        else:
            num_examples_per_epoch = self._num_examples_per_epoch_for_eval
            filenames = [filepath]

        with tf.variable_scope("{}_input".format(input_type)):
            filename_queue = tf.train.string_input_producer(filenames)
            read_input = self._read(filename_queue)
            min_fraction_of_examples_in_queue = 0.4
            min_queue_examples = int(num_examples_per_epoch *
                                     min_fraction_of_examples_in_queue)

        return build_batch(image=read_input["feature"],
                           label=read_input["feature"],
                           min_queue_examples=min_queue_examples,
                           batch_size=batch_size,
                           shuffle=(input_type == InputType.train))

    def _read(self, filename_queue):
        result = {
            "label_len": None,
            "label": None,
            "feature_len": None,
            "feature": None
        }
        image_depth = 1
        image_bytes = 32*32
        reader = tf.FixedLengthRecordReader(record_bytes=image_bytes)
        key, value = reader.read(filename_queue)
        record_byes = tf.decode_raw(value, tf.uint8)
        image = tf.reshape(
            tf.slice(record_byes, [0], [image_bytes]),
            [image_depth, 32, 32]
        )
        result["feature"] = tf.cast(
            tf.transpose(image, [1, 2, 0]),
            tf.float32
        )
        image = tf.divide(result["feature"], 255.0)
        result["feature"] = scale_image(image)
        return result