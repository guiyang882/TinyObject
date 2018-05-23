# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import codecs
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


dir_prefix = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/"
sign_label_path = dir_prefix + "sub_traffic_sign.json"
annotation_path = dir_prefix + "data/annotations.json"
images_dir = dir_prefix + "data/"
standard_images_dir = dir_prefix + "standard_data/"
target_train_dir = dir_prefix + "standard_data/train/"
target_test_dir = dir_prefix + "standard_data/test/"

target_img_voc_dir = dir_prefix + "VOCFORMAT/JPEGImages/"
target_annotation_voc_dir = dir_prefix + "VOCFORMAT/Annotations/"

SSD_IMG_W, SSD_IMG_H = 512, 512
RANDOM_SAMPLE_NUM = 500  # 随机采样500张原始图像用来生成训练样本

with open(sign_label_path, "r") as handler:
    sign_label = json.load(handler)
id2sign_dict = dict()
for sign, nums in sign_label.items():
    id2sign_dict[nums] = sign
print(id2sign_dict)

def load_annotation(is_train=True):
    def _fetch_image_label_pos(img_detail):
        sub_anno_info = list()
        for cell in img_detail["objects"]:
            if cell["category"] in sign_label.keys():
                tmp_label_info = list()
                tmp_label_info.append(round(cell["bbox"]["xmin"]))
                tmp_label_info.append(round(cell["bbox"]["ymin"]))
                tmp_label_info.append(round(cell["bbox"]["xmax"]))
                tmp_label_info.append(round(cell["bbox"]["ymax"]))
                tmp_label_info.append(sign_label[cell["category"]])
                is_valid = True
                for item in tmp_label_info:
                    if item <= 0:
                        is_valid = False
                if is_valid:
                    sub_anno_info.append(tmp_label_info)
        return sub_anno_info

    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
        for img_name, img_detail in annotation_info["imgs"].items():
            str_img_type = img_detail["path"].split("/")[0]
            if img_detail["objects"]:
                sub_anno_info = _fetch_image_label_pos(img_detail)
            else:
                sub_anno_info = list()

            if not sub_anno_info:
                continue
            img_path = images_dir + img_detail["path"]
            src_image = cv2.imread(img_path)
            print(sub_anno_info)
            create_ssd_training_samples(src_image, sub_anno_info, img_name, is_train)
            # break


# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples(
        src_image, anno_targets, image_name, much_signs, need_signs, is_train=True):
    # 保存裁剪出来的结果
    if is_train:
        if not os.path.exists(target_train_dir):
            os.makedirs(target_train_dir)
        if not os.path.exists(target_train_dir + "src"):
            os.makedirs(target_train_dir + "src")
        if not os.path.exists(target_train_dir + "annotation"):
            os.makedirs(target_train_dir + "annotation")
    else:
        if not os.path.exists(target_test_dir):
            os.makedirs(target_test_dir)
        if not os.path.exists(target_test_dir + "src"):
            os.makedirs(target_test_dir + "src")
        if not os.path.exists(target_test_dir + "annotation"):
            os.makedirs(target_test_dir + "annotation")

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

    def _random_crop_for_target():
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        for idx in range(0, len(anno_targets)):
            c_x = (anno_targets[idx][0] + anno_targets[idx][2]) // 2
            c_y = (anno_targets[idx][1] + anno_targets[idx][3]) // 2

            u_x = random.randint(max(0, c_x - SSD_IMG_W // 2), anno_targets[idx][0])
            u_y = random.randint(max(0, c_y - SSD_IMG_H // 2), anno_targets[idx][1])

            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            print(area)
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        return crop_list, anno_list

    def _random_crop_for_target_with_condition(much_signs, need_signs):
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        for idx in range(0, len(anno_targets)):
            if anno_targets[idx][-1] in need_signs:
                c_x = (anno_targets[idx][0] + anno_targets[idx][2]) // 2
                c_y = (anno_targets[idx][1] + anno_targets[idx][3]) // 2

                u_x = random.randint(max(0, c_x - SSD_IMG_W // 2), anno_targets[idx][0])
                u_y = random.randint(max(0, c_y - SSD_IMG_H // 2), anno_targets[idx][1])

                area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
                # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
                trans_targets = _crop_valid(area, anno_targets)
                is_find = True
                for cell in trans_targets:
                    if cell[-1] in much_signs:
                        is_find = False
                if trans_targets and is_find:
                    crop_list.append(area)
                    anno_list.append(trans_targets)
        return crop_list, anno_list

    crop_list, anno_list = _random_crop_for_target_with_condition(much_signs,
                                                                  need_signs)
    for i in range(len(crop_list)):
        x0, y0, x1, y1 = crop_list[i]
        # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
        sub_img = src_image[y0:y1, x0:x1]
        f_name = image_name + "_%d_%d_%d_%d_%d.png" % (x0, y0, x1, y1, i)
        cv2.imwrite(target_img_voc_dir + f_name, sub_img)
        f_name = image_name + "_%d_%d_%d_%d_%d.json" % (x0, y0, x1, y1, i)
        with codecs.open(target_annotation_voc_dir + f_name, "w") as handle:
            handle.write(json.dumps(anno_list[i]))
        print(target_annotation_voc_dir + f_name)
    print("after deal with ", image_name)


def show_distribution():
    anno_list = os.listdir(target_annotation_voc_dir)
    sign_distribution_dict = dict()
    for anno_file in anno_list:
        abs_anno_path = target_annotation_voc_dir + anno_file
        with open(abs_anno_path, "r") as handler:
            target_list = json.load(handler)
            for target in target_list:
                sign_distribution_dict[target[-1]] = \
                    sign_distribution_dict.get(target[-1], 0) + 1
    print(sign_distribution_dict)
    sign_list, sign_cnt = list(), list()
    for key, val in sign_distribution_dict.items():
        sign_list.append(key)
        sign_cnt.append(val)
    plt.bar(sign_list, sign_cnt, width=0.3, color='y')
    plt.show()

def remove_file(sign_anno_name):
    for sign_name in sign_anno_name:
        os.remove(target_annotation_voc_dir + sign_name)
        img_name = ".".join(sign_name.split(".")[:-1] + ["png"])
        os.remove(target_img_voc_dir + img_name)

def find_sign_id():
    anno_list = os.listdir(target_annotation_voc_dir)
    sign_distribution_dict = dict()
    for anno_file in anno_list:
        abs_anno_path = target_annotation_voc_dir + anno_file
        with open(abs_anno_path, "r") as handler:
            target_list = json.load(handler)
            is_valid = False
            for target in target_list:
                if target[-1] in [29]:
                    is_valid = True
            if is_valid:
                for target in target_list:
                    sign_distribution_dict[target[-1]] = \
                        sign_distribution_dict.get(target[-1], 0) + 1
    sign_list, sign_cnt = list(), list()
    for key, val in sign_distribution_dict.items():
        sign_list.append(key)
        sign_cnt.append(val)
    plt.bar(sign_list, sign_cnt, width=0.3, color='y')
    plt.show()
#     remove_file(sign_images)
find_sign_id()

def clear_sign_id():
    anno_list = os.listdir(target_annotation_voc_dir)
    sign_images = list()
    for anno_file in anno_list:
        abs_anno_path = target_annotation_voc_dir + anno_file
        with open(abs_anno_path, "r") as handler:
            target_list = json.load(handler)
            is_valid = False
            for target in target_list:
                if target[-1] in [29, 22, 51, 53]:
                    is_valid = True
                else:
                    is_valid = False
                    break
            if is_valid:
                sign_images.append(anno_file)
    remove_file(sign_images)

clear_sign_id()


def aug_sign_id():
    much_signs = [4, 6, 9, 22, 29, 30, 37, 47, 55, 58]
    need_signs = [1, 8, 11, 12, 13, 14, 15, 23, 24, 25, 31, 32, 33]
    anno_list = os.listdir(target_annotation_voc_dir)
    sign_images = list()
    for anno_file in anno_list:
        abs_anno_path = target_annotation_voc_dir + anno_file
        with open(abs_anno_path, "r") as handler:
            target_list = json.load(handler)
            is_valid = False
            for target in target_list:
                if target[-1] in need_signs:
                    is_valid = True
                elif target[-1] in much_signs:
                    is_valid = False
                    break
            if is_valid:
                img_name = anno_file.split("_")[0]
                sign_images.append(img_name)
    print(len(sign_images))
    # should augimg_path
    def _fetch_image_label_pos(img_detail):
        sub_anno_info = list()
        for cell in img_detail["objects"]:
            if cell["category"] in sign_label.keys():
                tmp_label_info = list()
                tmp_label_info.append(round(cell["bbox"]["xmin"]))
                tmp_label_info.append(round(cell["bbox"]["ymin"]))
                tmp_label_info.append(round(cell["bbox"]["xmax"]))
                tmp_label_info.append(round(cell["bbox"]["ymax"]))
                tmp_label_info.append(sign_label[cell["category"]])
                is_valid = True
                for item in tmp_label_info:
                    if item <= 0:
                        is_valid = False
                if is_valid:
                    sub_anno_info.append(tmp_label_info)
        return sub_anno_info

    sign_images_set = set(sign_images)
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
    for img_name in sign_images_set:
        part_path = annotation_info["imgs"][img_name]["path"]
        abs_img_path = images_dir + part_path
        src_image = cv2.imread(abs_img_path)
        sub_anno_info = _fetch_image_label_pos(annotation_info["imgs"][img_name])
        for i in range(0, 50):
            create_ssd_training_samples(src_image, sub_anno_info, img_name,
                                        much_signs, need_signs, True)

if __name__ == "__main__":
    load_annotation()