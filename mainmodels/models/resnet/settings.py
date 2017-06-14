# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ResNetConfig(object):
    dataset = "cifar10"
    mode = "train"
    train_data_path = "/Volumes/projects/TrainData/CIFAR/cifar-10-batches-bin"
    eval_data_path = "/Volumes/projects/TrainData/CIFAR/cifar-10-batches-bin"
    image_size = 32
    log_base_dir = "/Users/liuguiyang/Documents/CodeProj/PyProj/TinyObject" \
                   "/mainmodels/log/resnet/"
    train_log_dir = log_base_dir + "train"
    eval_log_dir = log_base_dir + "eval"
    num_gpus = 0
    eval_once = False
    eval_batch_count = 50

g_ResNetConfig = ResNetConfig()