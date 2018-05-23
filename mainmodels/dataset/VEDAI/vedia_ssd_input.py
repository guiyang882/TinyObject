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
from mainmodels.dataset.VEDAI.tools import fetch_xml_format


vedia_voc_dir = "/Volumes/projects/repos/RSI/VEDAI/VEDIAVOCFORAMT/"
vedia_voc_image_dir = vedia_voc_dir + "JPEGImages/"
vedia_voc_anno_dir = vedia_voc_dir + "Annotations/"

base_dir = "/Volumes/projects/repos/RSI/VEDAI/"
img_512_dir = base_dir + "512/Vehicules512/"
anno_512_dir = base_dir + "512/Annotations512/"


# 将vedia的原始数据转换成ssd模型voc格式的数据输入
def create_ssd_samples(img_dir, anno_dir):
    if not os.path.exists(vedia_voc_image_dir):
        os.makedirs(vedia_voc_image_dir)
    if not os.path.exists(vedia_voc_anno_dir):
        os.makedirs(vedia_voc_anno_dir)

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
        xml_obj = fetch_xml_format(src_img, img_name.replace("png", "jpg"),
                                   normal_anno_targets)
        # 将图像和标注文件拷贝到指定的位置
        with open(vedia_voc_anno_dir+img_name.replace("png", "xml"), "w") as \
                anno_writer:
            anno_writer.write(xml_obj)
        cv2.imwrite(vedia_voc_image_dir+img_name.replace("png", "jpg"),
                    src_img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
        print(img_name)


if __name__ == '__main__':
    create_ssd_samples(img_512_dir, anno_512_dir)