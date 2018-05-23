# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile

from mainmodels.dataset.NWPUVHR10 import tools

dir_prefix = "/Volumes/projects/repos/RSI/NWPUVHR10/"
annotation_dir = dir_prefix + "sub_annotation/"
image_dir = dir_prefix + "large_src/"
nwpu_voc_dir = "/Volumes/projects/repos/RSI/NWPUVHR10/NWPUVOCFORMAT/"
nwpu_voc_image_dir = nwpu_voc_dir + "JPEGImages/"
nwpu_voc_anno_dir = nwpu_voc_dir + "Annotations/"

# 将NWPU的标注文件转换成VOC对应的xml文件
def convert2vocxml(src_image, img_name,abs_anno_path):
    total_annos = tools.fetch_anno_targets_info(
        abs_anno_path, is_label_text=True)
    return tools.fetch_xml_format(src_image, img_name, total_annos)


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

# 分析下原始数据的图像打下分布
def dist_images_info(is_show=False, is_format=False):
    image_areas = list()
    for img_name in os.listdir(image_dir):
        if img_name.startswith("._"):
            continue
        src_img = cv2.imread(image_dir+img_name)
        h, w = src_img.shape[:2]
        image_areas.append([h, w])
    print(image_areas)
    if is_show:
        img_np = np.array(image_areas)
        plt.grid(True)
        plt.scatter(img_np[:, 0], img_np[:, 1])
        plt.show()
    if is_format:
        for img_name in os.listdir(image_dir):
            if img_name.startswith("._"):
                continue
            src_img = cv2.imread(image_dir+img_name)
            h, w = src_img.shape[:2]
            h_w_ratio = h / w
            if h_w_ratio >= 3/4 and h_w_ratio <= 4/3:
                # 拷贝相应的文件到指定的目录，同时生成对应的xml文件
                cv2.imwrite(nwpu_voc_image_dir+img_name, src_img)
                xml_obj = convert2vocxml(
                    src_img, img_name, annotation_dir+img_name.replace("jpg", "txt"))
                with open(nwpu_voc_anno_dir+img_name.replace("jpg", "xml"),
                          "w") as xml_writer:
                    xml_writer.write(xml_obj)
                continue
            abs_anno_path = annotation_dir + img_name.replace("jpg", "txt")
            anno_targets = tools.fetch_anno_targets_info(
                abs_anno_path, is_label_text=True)
            # 对于不在这个比例范围的图像，我们通过裁剪将图像适应相应的尺度
            if h > w: CROP_IMG_W, CROP_IMG_H = w, int(w / 3 * 4)
            else: CROP_IMG_W, CROP_IMG_H = int(h/3*4), h
            crop_list, anno_list = [], []
            for lx in range(0, w, int(CROP_IMG_W // 2.5)):
                for ly in range(0, h, int(CROP_IMG_H // 2.5)):
                    u_x, u_y = lx, ly
                    if lx + CROP_IMG_W > w:
                        u_x = w - CROP_IMG_W
                    if ly + CROP_IMG_H > h:
                        u_y = h - CROP_IMG_H
                    area = [u_x, u_y, u_x + CROP_IMG_W, u_y + CROP_IMG_H]
                    trans_targets = _crop_valid(area, anno_targets)
                    if trans_targets:
                        crop_list.append(area)
                        anno_list.append(trans_targets)
            crop_stats_set = set()
            for i in range(len(crop_list)):
                x0, y0, x1, y1 = crop_list[i]
                f_name_prefix = img_name[:-4] + "_%d_%d_%d_%d.jpg" % (
                    x0, y0, x1, y1)
                if f_name_prefix not in crop_stats_set:
                    crop_stats_set.add(f_name_prefix)
                else:
                    continue
                # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
                sub_img = src_img[y0:y1, x0:x1]
                f_name = img_name[:-4] + "_%d_%d_%d_%d_%d.jpg" % (
                    x0, y0, x1, y1, i)
                cv2.imwrite(nwpu_voc_image_dir + f_name, sub_img)
                xml_obj = tools.fetch_xml_format(sub_img, f_name, anno_list[i])
                f_name = img_name[:-4] + "_%d_%d_%d_%d_%d.xml" % (
                    x0, y0, x1, y1, i)
                with open(nwpu_voc_anno_dir + f_name, "w") as handle:
                    handle.write(xml_obj)
                print(f_name)
    image_areas = list()
    for img_name in os.listdir(nwpu_voc_image_dir):
        if img_name.startswith("._"):
            continue
        src_img = cv2.imread(nwpu_voc_image_dir + img_name)
        h, w = src_img.shape[:2]
        image_areas.append([h, w])
    print(image_areas)
    if is_show:
        img_np = np.array(image_areas)
        plt.grid(True)
        plt.scatter(img_np[:, 0], img_np[:, 1])
        plt.show()

# 每个类在数据中的分布的情况
def each_class_distribution(is_show=False):
    class_nums = dict()
    class_areas = dict()
    for anno_name in os.listdir(annotation_dir):
        if anno_name.startswith("._"):
            continue
        abs_anno_path = annotation_dir + anno_name
        total_annos = tools.fetch_anno_targets_info(abs_anno_path)
        for item in total_annos:
            if item[-1] not in class_nums:
                class_nums[item[-1]] = 1
            else:
                class_nums[item[-1]] += 1
            if item[-1] not in class_areas:
                class_areas[item[-1]] = list()
            class_areas[item[-1]].append([item[2]-item[0], item[3]-item[1]])
    # print(class_nums)
    # print(class_areas)
    if is_show:
        pass


if __name__ == '__main__':
    # dist_images_info(is_show=False, is_format=False)
    tools.show_targets(nwpu_voc_image_dir, nwpu_voc_anno_dir)