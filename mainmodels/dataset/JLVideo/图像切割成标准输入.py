# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/11

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 把吉林一号视频卫星数据切割成指定大小的数据进行批量的检测
import os
import json
import codecs
import random
import cv2
import time
from lxml.etree import Element, SubElement, tostring
from mainmodels.dataset import tools


dir_prefix = "/Volumes/projects/repos/RSI/CSUVOCFormat/"
tpl_src_annotation_dir = dir_prefix + "source/{}/Annotations/"
tpl_src_images_dir = dir_prefix + "source/{}/JPEGImages/"


SSD_IMG_W, SSD_IMG_H = 608, 608

tpl_img_voc_dir = dir_prefix + "{}x{}/{}/JPEGImages/"
tpl_annotation_voc_dir = dir_prefix + "{}x{}/{}/Annotations/"
tpl_src_png_dir = dir_prefix + "{}x{}/{}/PNGImages/"

sign_label = {
    "plane": 1
}

id2sign_dict = {
    1: "plane"
}

def load_annotation(video_name):
    src_annotation_dir = tpl_src_annotation_dir.format(video_name)
    src_images_dir = tpl_src_images_dir.format(video_name)
    for idx in range(1, len(os.listdir(src_annotation_dir))+1, 5):
        anno_name = "%06d.xml" % idx
        src_img_name = "%06d.jpg" % idx
        anno_path = src_annotation_dir + anno_name
        img_path = src_images_dir + src_img_name
        src_image = cv2.imread(img_path)
        targets_info = tools.extract_airplane_posinfo(anno_path)
        if len(targets_info) == 0:
            continue
        create_ssd_training_samples(video_name, src_image, targets_info, src_img_name)

# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples(
        video_name, src_image, anno_targets, image_name):
    # 保存裁剪出来的结果
    def _crop_valid(area, anno_targets):
        anno_res = []
        for info in anno_targets:
            if ((info[0] >= area[0] and info[1] >= area[1]) and
                (info[2] <= area[2] and info[3] <= area[3])):
                anno_res.append(
                    [info[0] - area[0], info[1] - area[1],
                     info[2] - area[0], info[3] - area[1],
                     info[-1]])
            if (info[0] >= area[0] and info[1] >= area[1] and
                info[0] < area[2] and info[1] < area[3] and
                (not (info[2] <= area[2] and info[3] <= area[3]))):
                base = (info[2] - info[0]) * (info[3] - info[1])
                x_max_min = min(info[2], area[2])
                y_max_min = min(info[3], area[3])
                new_square = (x_max_min - info[0]) * (y_max_min - info[1])
                if new_square / base >= 0.75:
                    anno_res.append(
                        [info[0] - area[0], info[1] - area[1],
                         x_max_min - area[0], y_max_min - area[1],
                         info[-1]])
        return anno_res

    def _align_crop_for_target():
        h, w = src_image.shape[:2]
        crop_list, anno_list = [], []
        for lx in range(0, w, int(SSD_IMG_W//1)):
            for ly in range(0, h, int(SSD_IMG_H//1)):
                u_x, u_y = lx, ly
                if lx + SSD_IMG_W > w:
                    u_x = w - SSD_IMG_W
                if ly + SSD_IMG_H > h:
                    u_y = h - SSD_IMG_H
                area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
                trans_targets = _crop_valid(area, anno_targets)
                if trans_targets:
                    crop_list.append(area)
                    anno_list.append(trans_targets)
        return crop_list, anno_list

    crop_list, anno_list = _align_crop_for_target()
    target_img_voc_dir = tpl_img_voc_dir.format(
        SSD_IMG_W, SSD_IMG_H, video_name)
    target_annotation_voc_dir = tpl_annotation_voc_dir.format(
        SSD_IMG_W, SSD_IMG_H, video_name)
    if not os.path.exists(target_img_voc_dir):
        os.makedirs(target_img_voc_dir)
    if not os.path.exists(target_annotation_voc_dir):
        os.makedirs(target_annotation_voc_dir)

    image_name = ".".join(image_name.split(".")[:-1])
    for i in range(len(crop_list)):
        x0, y0, x1, y1 = crop_list[i]
        # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
        sub_img = src_image[y0:y1, x0:x1]
        f_name = image_name + "_%d_%d_%d_%d_%d.jpg" % (x0, y0, x1, y1, i)
        cv2.imwrite(target_img_voc_dir + f_name, sub_img)
        xml_obj = tools.fetch_xml_format(sub_img, f_name, anno_list[i])
        f_name = image_name + "_%d_%d_%d_%d_%d.xml" % (x0, y0, x1, y1, i)
        with codecs.open(target_annotation_voc_dir + f_name, "w") as handle:
            handle.write(xml_obj)
        print(target_annotation_voc_dir + f_name)


if __name__ == '__main__':
    load_annotation("large_000013363_total")