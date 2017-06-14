# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-4])

class RpnPlusModelConfig(object):
    vgg16_model_path = "/".join([proj_dir, "model", "vgg16.npy"])
    rpnplus_model_path = "/".join([proj_dir, "model", "rpnplus_model.npy"])

    save_rpnplus_model_dir = "/".join([proj_dir, "model"])

    image_input_dir = "/".join([proj_dir, "trainmodel", "data", "images"])
    image_save_dir = "/".join([proj_dir, "output"])

    # 主要是用来存储图像和对应的标签的位置的文件
    train_samples_index_path = \
        "/Users/liuguiyang/Downloads/AirplaneSamples/Positive/train/index.txt"
    test_samples_index_path = \
        "/Users/liuguiyang/Downloads/AirplaneSamples/Positive/test/index.txt"

    image_height = 224
    image_width = 224
    image_depth = 1

    feature_ratio = 8

    batch_size = 256

g_RpnPlus_Config = RpnPlusModelConfig()
