# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

dir_prefix = "/root/caffe/data/LSD10/"

img_voc_dir = dir_prefix + "JPEGImages/"
annotation_voc_dir = dir_prefix + "Annotations/"


def organizing_model_data(from_file_path, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    writer = open(save_path, "w")
    with open(from_file_path, "r") as reader:
        for line in reader:
            img_name = line.strip()
            abs_img_path = img_voc_dir + img_name
            abs_anno_path = annotation_voc_dir + img_name.replace("jpg", "xml")
            writer.write("{} {}\n".format(abs_img_path, abs_anno_path))




if __name__ == '__main__':
    from_file_path = "/root/caffe/data/LSD10/all_train.txt"
    save_path = "/root/caffe/data/LSD10/train.txt"
    organizing_model_data(from_file_path, save_path)