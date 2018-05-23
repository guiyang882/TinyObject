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
import pprint
import numpy as np


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


def standard_calc():
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)

    sign_nums_dict = dict()
    for _, img_detail in annotation_info["imgs"].items():
        for cell in img_detail["objects"]:
            if cell["category"] not in sign_nums_dict:
                sign_nums_dict[cell["category"]] = 1
            else:
                sign_nums_dict[cell["category"]] += 1
    sign_list = [(key, val) for key, val in sign_nums_dict.items()]
    sign_list = sorted(sign_list, key=lambda x: x[1], reverse=True)
    too_much_list = sign_list[:14]  #太多的标记，最好不要做增强
    need_list = sign_list[14:60]  #需要做增强的数据
    too_much_dict, need_dict = dict(), dict()
    for cell in too_much_list:
        too_much_dict[cell[0]] = cell[1]
    for cell in need_list:
        need_dict[cell[0]] = cell[1]
    std_list = [sign_list[i][1] for i in range(60)]
    mean_val = np.mean(std_list)
    '''
    如何做数据的均衡：
    1. 先找到包含need_list且不包含too_much的类别的图片
    2. 一轮的一轮进行随机生成，每轮过后，统计每个类别的数量，再决定是否进行下一轮
    3. 达到要求后退出即可
    '''
    while True:
        should_aug_list = list()
        for img_name, img_detail in annotation_info["imgs"].items():
            for cell in img_detail["objects"]:
                if cell["category"] in too_much_dict:
                    continue
                if cell["category"] in need_dict:
                    should_aug_list.append(img_detail["path"])
                    sign_nums_dict[cell["category"]] = mean_val

        sign_list = [(key, val) for key, val in sign_nums_dict.items()]
        sign_list = sorted(sign_list, key=lambda x: x[1], reverse=True)
        too_much_list = sign_list[:14]  # 太多的标记，最好不要做增强
        need_list = sign_list[14:60]  # 需要做增强的数据
        too_much_dict, need_dict = dict(), dict()
        for cell in too_much_list:
            too_much_dict[cell[0]] = cell[1]
        for cell in need_list:
            need_dict[cell[0]] = cell[1]
        # 保证前60个元素的方差在
        std_list = [sign_list[i][1] for i in range(60)]
        mean_val = np.mean(std_list)
#         print(np.std(std_list))
#         print(std_list)
        if np.std(std_list) < 10:
            break
    pprint.pprint(too_much_dict)
    pprint.pprint(need_dict)


too_much_dict = {
    'il100': 2919.3121982526281,
    'io': 2919.3121982526281,
    'p10': 2919.3121982526281,
    'p11': 2919.3121982526281,
    'pa14': 2919.3121982526281,
    'ph4': 2919.3121982526281,
    'pl20': 2919.3121982526281,
    'pl70': 2919.3121982526281,
    'pl90': 2919.3121982526281,
    'pm20': 2919.3121982526281,
    'pn': 2997,
    'po': 2919.3121982526281,
    'ps': 2919.3121982526281,
    'w63': 2919.3121982526281
}

need_dict = {
    'i10': 2918.2358933384862,
    'i2': 2919.3121982526281,
    'i4': 2919.3121982526281,
    'i5': 2918.2358933384862,
    'il60': 2918.2358933384862,
    'il80': 2919.3121982526281,
    'il90': 2919.3121982526281,
    'ip': 2919.3121982526281,
    'p1': 2919.3121982526281,
    'p12': 2919.3121982526281,
    'p19': 2919.3121982526281,
    'p23': 2919.3121982526281,
    'p26': 2919.3121982526281,
    'p27': 2918.2358933384862,
    'p3': 2918.2358933384862,
    'p5': 2918.2358933384862,
    'p6': 2918.2358933384862,
    'p9': 2919.3121982526281,
    'pb': 2919.3121982526281,
    'pg': 2919.3121982526281,
    'ph4.5': 2919.3121982526281,
    'ph5': 2919.3121982526281,
    'pl100': 2919.3121982526281,
    'pl120': 2918.2358933384862,
    'pl15': 2919.3121982526281,
    'pl30': 2919.3121982526281,
    'pl40': 2919.3121982526281,
    'pl5': 2918.2358933384862,
    'pl50': 2919.3121982526281,
    'pl60': 2919.3121982526281,
    'pl80': 2919.3121982526281,
    'pm30': 2918.2358933384862,
    'pm55': 2919.3121982526281,
    'pne': 2919.3121982526281,
    'pr40': 2919.3121982526281,
    'pr60': 2918.2358933384862,
    'w13': 2919.3121982526281,
    'w21': 2918.2358933384862,
    'w22': 2919.3121982526281,
    'w30': 2919.3121982526281,
    'w32': 2919.3121982526281,
    'w55': 2919.3121982526281,
    'w57': 2918.2358933384862,
    'w58': 2919.3121982526281,
    'w59': 2919.3121982526281,
    'wo': 2919.3121982526281
}

selected_sign_dict = dict()
for key, val in too_much_dict.items():
    selected_sign_dict[key] = int(val)
for key, val in need_dict.items():
    selected_sign_dict[key] = int(val)

# 找到每个sign大约需要增加到多少才能保证数据的近似均衡
# 现在我们要对每张图片中含有sign的标注进行统计，确定大约需要从不同的几张图像中做多少增强的图
def cal_each_sign_distribution():
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)

    sign_nums_dict = dict()
    for _, img_detail in annotation_info["imgs"].items():
        for cell in img_detail["objects"]:
            if cell["category"] in too_much_dict or cell["category"] in need_dict:
                if cell["category"] not in sign_nums_dict:
                    sign_nums_dict[cell["category"]] = {
                        img_detail["path"]: 1,
                        "total": 0
                    }
                else:
                    if img_detail["path"] in sign_nums_dict[cell["category"]]:
                        sign_nums_dict[cell["category"]][img_detail["path"]] += 1
                    else:
                        sign_nums_dict[cell["category"]][img_detail["path"]] = 1
                sign_nums_dict[cell["category"]]["total"] += 1
    return sign_nums_dict
sign_nums_dict = cal_each_sign_distribution()
print(sign_nums_dict)

with open(sign_label_path, "r") as handler:
    sign_label = json.load(handler)
id2sign_dict = dict()
for sign, nums in sign_label.items():
    id2sign_dict[nums] = sign
print(id2sign_dict)

def find_aug_in_each_image():

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
                sub_anno_info.append(tmp_label_info)
        return sub_anno_info

    # 将图像中对应的label数据保证均衡，然后根据产生的sign_nums_dict进行图像的生成
    # 我们得到的数据格式如下
#     '''
#     label01: {
#         "image_path": cnt,
#         "image_path": cnt,
#         "image_path": cnt
#     }
#     '''
    for sign, sign_detail in sign_nums_dict.items():
        total_cnt = selected_sign_dict[sign]
        need_cnt = total_cnt - sign_detail["total"]
        while need_cnt:
            for key, val in sign_detail.items():
                if key == "total":
                    continue
                sign_nums_dict[sign][key] += 1
                sign_nums_dict[sign]["total"] += 1
                need_cnt -= 1
                if need_cnt == 0:
                    break
    # 我们需要将目标转换成如下格式
#     '''
#     image_path: {
#         "label01": cnt,
#         "label02": cnt,
#         "label03": cnt,
#     }
#     '''
    image2label_dict = dict()
    for sign, sign_detail in sign_nums_dict.items():
        for img_path, need_cnt in sign_detail.items():
            if img_path in image2label_dict:
                if sign in image2label_dict[img_path]:
                    image2label_dict[img_path][sign] += need_cnt
                else:
                    image2label_dict[img_path][sign] = need_cnt
            else:
                image2label_dict[img_path] = dict()
                image2label_dict[img_path][sign] = need_cnt
    print(image2label_dict)

    # 根据每一张图像的配置，进行图像数据的裁剪
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
    for img_path, sign_conf in image2label_dict.items():
        img_abs_path = images_dir + img_path
        src_image = cv2.imread(img_abs_path)
        img_name = img_path.strip().split("/")[-1].split(".")[0]
        img_detail = annotation_info["imgs"][img_name]
        sub_anno_info = _fetch_image_label_pos(img_detail)
        corp_subImg_by_configure(src_image, sub_anno_info, sign_conf, img_name)


def corp_subImg_by_configure(src_image, anno_targets, sign_conf, image_name):
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
        for idx in range(0, len(anno_list)):
            sign_id = anno_list[idx][-1]
            sign_name = id2sign_dict[sign_id]
            if sign_name not in sign_conf:
                continue
            if sign_conf[sign_name] <= 0:
                continue

            c_x = (anno_list[idx][0] + anno_list[idx][2]) // 2
            c_y = (anno_list[idx][1] + anno_list[idx][3]) // 2

            u_x = random.randint(max(0, c_x - SSD_IMG_W // 2),
                                 min(c_x + SSD_IMG_W // 2, img_width))
            u_y = random.randint(max(0, c_y - SSD_IMG_H // 2),
                                 min(c_y + SSD_IMG_H // 2, img_width))

            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        return crop_list, anno_list

    crop_list, anno_list = _random_crop_for_target()
    for i in range(len(crop_list)):
        x0, y0, x1, y1 = crop_list[i]
        # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
        sub_img = src_image[y0:y1, x0:x1]
        f_name = image_name + "_%d_%d_%d_%d_%d.png" % (x0, y0, x1, y1, i)
        cv2.imwrite(target_img_voc_dir + f_name, sub_img)
        f_name = image_name + "_%d_%d_%d_%d_%d.json" % (x0, y0, x1, y1, i)
        with codecs.open(target_annotation_voc_dir + f_name, "w") as handle:
            handle.write(json.dumps(anno_list[i]))


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
            create_ssd_training_samples(src_image, sub_anno_info, img_name, is_train)


# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples(
        src_image, anno_targets, image_name, is_train=True):
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

    def _random_crop():
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        while len(crop_list) < 10:
            u_x = random.randint(0, img_width - SSD_IMG_W)
            u_y = random.randint(0, img_height - SSD_IMG_H)
            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        return crop_list, anno_list

    def _align_crop():
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        for h in range(0, img_height-SSD_IMG_H, SSD_IMG_H-100):
            for w in range(0, img_width-SSD_IMG_W, SSD_IMG_W-100):
                area = [w, h, w+SSD_IMG_W, h+SSD_IMG_H]
                # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
                trans_targets = _crop_valid(area, anno_targets)
                if trans_targets:
                    crop_list.append(area)
                    anno_list.append(trans_targets)
        return crop_list, anno_list

    def _random_crop_for_target():
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        for idx in range(0, len(anno_list)):
            c_x = (anno_list[idx][0] + anno_list[idx][2]) // 2
            c_y = (anno_list[idx][1] + anno_list[idx][3]) // 2

            u_x = random.randint(max(0, c_x - SSD_IMG_W // 2),
                                 min(c_x + SSD_IMG_W // 2, img_width))
            u_y = random.randint(max(0, c_y - SSD_IMG_H // 2),
                                 min(c_y + SSD_IMG_H // 2, img_width))

            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        return crop_list, anno_list

    crop_list, anno_list = _random_crop_for_target()
    for i in range(len(crop_list)):
        x0, y0, x1, y1 = crop_list[i]
        # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
        sub_img = src_image[y0:y1, x0:x1]
        f_name = image_name + "_%d_%d_%d_%d_%d.png" % (x0, y0, x1, y1, i)
        cv2.imwrite(target_img_voc_dir + f_name, sub_img)
        f_name = image_name + "_%d_%d_%d_%d_%d.json" % (x0, y0, x1, y1, i)
        with codecs.open(target_annotation_voc_dir + f_name, "w") as handle:
            handle.write(json.dumps(anno_list[i]))
    print("after deal with ", image_name)


if __name__ == "__main__":
    load_annotation()