# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from mainmodels.dataset.VEDAI.tools import fetch_targets_info


base_dir = "/Volumes/projects/repos/RSI/VEDAI/"
img_512_dir = base_dir + "512/Vehicules512/"
anno_512_dir = base_dir + "512/Annotations512/"


# 标注出给定的坐标下的目标区域
def annotate_targets(src_image, anno_targets, line_width=1):
    for info in anno_targets:
        if len(info) == 9:
            pre_x, pre_y = info[0], info[1]
            for i in range(2, 8, 2):
                x, y = info[i], info[i+1]
                cv2.line(src_image, (pre_x, pre_y), (x, y), (0, 0, 255), line_width)
                pre_x, pre_y = x, y
            cv2.line(src_image, (pre_x, pre_y), (info[0], info[1]), (0, 0, 255), line_width)
        elif len(info) == 5:
            xmin, ymin, xmax, ymax = info[:4]
            cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), line_width)
    return src_image

def show_targets(img_dir, anno_dir):
    for img_name in os.listdir(img_dir):
        if img_name.startswith("._"):
            continue
        img_type = img_name.split(".")[0].split("_")[1]
        if img_type == "ir":
            continue
        img_prefix = img_name.split("_")[0]
        abs_img_path = img_dir + img_name
        abs_anno_path = anno_dir + img_prefix + ".txt"
        anno_targets, normal_anno_targets = fetch_targets_info(abs_anno_path)
        src_img = cv2.imread(abs_img_path)
        src_img = annotate_targets(src_img, anno_targets)
        src_img = annotate_targets(src_img, normal_anno_targets)
        cv2.imshow("src", src_img)
        print(img_name)
        cv2.waitKey()


if __name__ == '__main__':
    show_targets(img_512_dir, anno_512_dir)