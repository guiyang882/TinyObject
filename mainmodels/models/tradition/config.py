# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

proj_root = "/".join(os.path.abspath(__file__).split("/")[:-4])

class DataSet:
    positive_nums = 10
    negative_nums = 10
    class_nums = 2
    class_labels = [0, 1]

class BaseConfig(object):
    train_samplesp_path = "/Users/liuguiyang/Downloads/AirplaneSamples/input/train.tfrecords"
    test_samples_path = "/Users/liuguiyang/Downloads/AirplaneSamples/input/test.tfrecords"

    # image info
    image_width = 48
    image_height = 48
    image_depth = 1

    sample_ratio = {
        "train": 0.7,
        "test": 0.2,
        "valid": 0.1
    }
    batch_size = 96
    NUM_CLASSES = 2
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2986
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 747
    labels = {
        "neg": 0,
        "pos": 1
    }

    # Constants describing the training process.
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

    use_fp16 = False

    dataset = DataSet()

class CNNConfig(BaseConfig):
    train_log_dir = "/".join([proj_root, "trainmodel", "log", "cnn", "train"])
    eval_log_dir = "/".join([proj_root, "trainmodel", "log", "cnn", "eval"])

class MultiCNNConfig(BaseConfig):
    train_log_dir = "/".join([proj_root, "trainmodel", "log", "multicnn",
                              "train"])
    eval_log_dir = "/".join([proj_root, "trainmodel", "log", "multicnn",
                             "eval"])

g_CNNConfig = CNNConfig()
g_MultiCNNConfig = MultiCNNConfig()
