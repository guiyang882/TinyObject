# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/8/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, codecs, json
import cv2

from mainmodels.models.ssd.settings import g_SSDConfig


dir_prefix = g_SSDConfig.DATASET_BASE_DIR
if not dir_prefix.endswith("/"):
    dir_prefix += "/"


def show_src_dir():
    for img_file_name in os.listdir(dir_prefix + "src/"):
        if img_file_name.startswith("._"):
            continue
        print(img_file_name)
        anno_file = ".".join(img_file_name.split(".")[:-1] + ["json"])
        print(anno_file)
        src_img = cv2.imread(dir_prefix+"src/"+img_file_name)
        with codecs.open(dir_prefix+"annotation/"+anno_file, "r") as handle:
            anno_targets = json.load(handle)
            for area in anno_targets:
                print(area)
                cv2.rectangle(src_img, (area[0], area[1]), (area[2], area[3]),
                              (0, 255, 0), 2)
        cv2.imshow("dsfdsf", src_img)
        cv2.waitKey()


# 展示原始的标注信息
def show_annotation_image_file(file_name):
    anno_file = ".".join(file_name.split(".")[:-1] + ["json"])
    print(anno_file)
    abs_image_path = dir_prefix + "total_src/" + file_name
    print(abs_image_path)
    src_img = cv2.imread(dir_prefix + "src/" + file_name)
    with codecs.open(dir_prefix + "annotation/" + anno_file, "r") as handle:
        anno_targets = json.load(handle)
        for area in anno_targets:
            cv2.rectangle(src_img, (area[0], area[1]), (area[2], area[3]),
                          (0, 0, 255), 2)
    cv2.imshow("GT", src_img)
    cv2.waitKey()

# 展示SSD构造训练样本时产生的候选框的目标
def show_ssd_prepare_boxes(file_name):
    ssd_box_name = ".".join(file_name.strip().split(".")[:-1]) + "_annotation.txt"
    ssd_box_path = dir_prefix + "ssd_label_box_src/" + ssd_box_name
    src_img = cv2.imread(dir_prefix + "src/" + file_name)
    with codecs.open(ssd_box_path, "r", "utf8") as handler:
        for line in handler:
            line = line.strip().split(',')
            box_pos = [int(float(i)) for i in line]
            cv2.rectangle(src_img, tuple(box_pos[:2]), tuple(box_pos[2:]),
                          (0, 255, 0), 1)
    cv2.imshow("SSD_BOX", src_img)
    cv2.waitKey()

if __name__ == '__main__':
    pass