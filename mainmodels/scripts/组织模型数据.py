# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

dir_prefix = "/root/repos/CSUVOCFormat/"
SSD_IMG_W, SSD_IMG_H = 512, 512

tpl_img_voc_dir = dir_prefix + "{}x{}/{}/JPEGImages/"
tpl_annotation_voc_dir = dir_prefix + "{}x{}/{}/Annotations/"


def organizing_model_data(video_names, save_path):
    if os.path.exists(save_path):
        os.remove(save_path)
    for video_name in video_names:
        img_voc_dir = tpl_img_voc_dir.format(SSD_IMG_W, SSD_IMG_H, video_name)
        annotation_voc_dir = tpl_annotation_voc_dir.format(
            SSD_IMG_W, SSD_IMG_H, video_name)
        with open(save_path, "a") as writer:
            for img_name in os.listdir(img_voc_dir):
                if not os.path.exists(
                                annotation_voc_dir+img_name.replace("jpg", "xml")):
                    continue
                writer.write("{} {}\n".format(
                    img_voc_dir+img_name,annotation_voc_dir+img_name.replace("jpg", "xml")))


if __name__ == '__main__':
    video_names = ["large_000013363_total", "large_000014631_total", "large_minneapolis_1_total"]
    save_path = "/root/caffe/data/CSUVIDEO/trainval.txt"
    organizing_model_data(video_names, save_path)