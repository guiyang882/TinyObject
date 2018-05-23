# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/9/30

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt


dir_prefix = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/"
sign_label_path = dir_prefix + "sub_traffic_sign.json"
annotation_path = dir_prefix + "data/annotations.json"

target_img_voc_dir = dir_prefix + "VOCFORMAT/JPEGImages/"
target_annotation_voc_dir = dir_prefix + "VOCFORMAT/Annotations/"
target_train_info_dir = dir_prefix + "VOCFORMAT/ImageSets/Main/"


# 切分TT100K的训练集和测试集
def split_train_test():
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
    train_list, test_list = list(), list()
    for img_name, img_detail in annotation_info["imgs"].items():
        sub_img_path = img_detail["path"]
        data_type = sub_img_path.split("/")[0]
        if data_type == "other" or data_type == "test":
            test_list.append(img_name)
        else:
            train_list.append(img_name)

    # 从准备好的数据中，找到对应的图像，通过名字确定是训练还是测试
    train_set, test_set = set(), set()
    target_list = os.listdir(target_img_voc_dir)
    for target in target_list:
        tmp_img = target.split("_")[0]
        if tmp_img in test_list:
            test_set.add(target)
        if tmp_img in train_list:
            train_set.add(target)

    # 将不同的数据写到不同的路径下
    with open(target_train_info_dir + "test.txt", "r") as handelr:
        for item in test_set:
            handler.write(item)
    with open(target_train_info_dir + "trainval.txt", "r") as handelr:
        for item in train_set:
            handler.write(item)

# split_train_test()

def show_test_distribution():
    sign_distribution_dict = dict()
    with open(target_train_info_dir+"test.txt", "r") as test_reader:
        for line in test_reader.readlines():
            line = line.strip()
            if len(line) <= 5:
                continue
            abs_anno_path = target_annotation_voc_dir + line + ".json"
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

def show_train_distribution():
    sign_distribution_dict = dict()
    with open(target_train_info_dir+"trainval.txt", "r") as test_reader:
        for line in test_reader.readlines():
            line = line.strip()
            if len(line) <= 5:
                continue
            abs_anno_path = target_annotation_voc_dir + line + ".json"
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

def calc_rgb_mean():
    r_list, g_list, b_list = list(), list(), list()
    with open(target_train_info_dir+"all.txt", "r") as reader:
        for line in reader.readlines():
            line = line.strip()
            if len(line) <= 5:
                continue
            abs_img_path = target_img_voc_dir + line + ".png"
            src_img = cv2.imread(abs_img_path)
            b, g, r = cv2.split(src_img)
            b_list.append(np.mean(b))
            g_list.append(np.mean(g))
            r_list.append(np.mean(r))
    print(np.mean(r_list))
    print(np.mean(g_list))
    print(np.mean(b_list))

# 展示原始数据的分布情况
def show_origin_signs_distribution():
    signs_dict = dict()
    annotation_path = "/Volumes/projects/repos/TT100K/annotations.json"
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
        for img_name, img_detail in annotation_info["imgs"].items():
            for item in img_detail["objects"]:
                signs_dict[item["category"]] = signs_dict.get(
                    item["category"], 0) + 1
    # print(signs_dict)
    x_label, y_label = list(), list()
    for key, val in signs_dict.items():
        print("%s,%d" % (key, val))
        x_label.append(key)
        y_label.append(val)
    plt.bar(x_label, y_label, width=0.3 , color='b')
    plt.show()
# show_origin_signs_distribution()

# 展示原始数据的分布情况
def show_selected_signs_distribution():
    signs_dict = dict()
    sub_signs_dict = dict()
    with open("/Volumes/projects/repos/TT100K/sub_traffic_sign.json", "r") as\
            reader:
        sub_signs_dict = json.load(reader)
    annotation_path = "/Volumes/projects/repos/TT100K/annotations.json"
    with open(annotation_path, "r") as handler:
        annotation_info = json.load(handler)
        for img_name, img_detail in annotation_info["imgs"].items():
            for item in img_detail["objects"]:
                if item["category"] not in sub_signs_dict:
                    continue
                signs_dict[item["category"]] = signs_dict.get(
                    item["category"], 0) + 1
    for key, val in signs_dict.items():
        print("%s,%d" % (key, val))
# show_selected_signs_distribution()

signs_dict = {
    1: 922, 2: 1891, 3: 2883, 4: 3514, 5: 1391, 6: 3041, 7: 1556,
    8: 1556, 9: 2002, 10: 1086, 11: 2542, 12: 2777, 13: 2963, 14: 1098,
    15: 986, 16: 1701, 17: 1812, 18: 2730, 19: 1398, 20: 1828, 21: 1231,
    22: 3502, 23: 3082, 24: 3985, 25: 3322, 26: 1422, 27: 1049, 28: 2950,
    29: 4794, 30: 3304, 31: 698, 32: 3755, 33: 3447, 34: 3352, 35: 2216,
    36: 1513, 37: 2101, 38: 1553, 39: 1888, 40: 1686, 41: 3406, 42: 1975,
    43: 2121, 44: 3485, 45: 2476, 46: 3352, 47: 2442, 48: 2056, 49: 1685,
    50: 2069, 51: 2605, 52: 3677, 53: 2225, 54: 2112, 55: 2364, 56: 2278,
    57: 2034, 58: 3199, 59: 3367, 60: 1596
}

# 展示原始数据的分布情况
def show_signs_distribution():
    idx_signs_dict = dict()
    with open("/Volumes/projects/repos/TT100K/sub_traffic_sign.json", "r") as\
            reader:
        sub_signs_dict = json.load(reader)
        for sign, idx in sub_signs_dict.items():
            idx_signs_dict[idx] = sign

    for key, val in signs_dict.items():
        print("%s,%d" % (idx_signs_dict[key], val))
# show_signs_distribution()


# 分析初不同的目标在不同图像中的位置写到不同的文件中
# comp4_det_test_{label}.txt
#   image_name_prefix prob x0 y0 x1 y1
#   image_name_prefix prob x0 y0 x1 y1
#   image_name_prefix prob x0 y0 x1 y1
def create_test_targets():
    test_file_path = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua" \
                     "/VOCFORMAT/ImageSets/Main/test.txt"
    test_path_list = list()
    with open(test_file_path, "r") as path_reader:
        for line in path_reader.readlines():
            line = line.strip()
            test_path_list.append(line + ".json")
    with open(sign_label_path, "r") as sign_reader:
        sign_idx_dict = json.load(sign_reader)
        idx_sign_dict = dict()
        for key, val in sign_idx_dict.items():
            idx_sign_dict[val] = key
    anno_prefix = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/VOCFORMAT" \
                  "/Annotations/"
    save_prefix = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/VOCFORMAT" \
                  "/ImageSets/Main/std_test/"
    save_template = save_prefix + "comp4_det_std_{0}.txt"
    for anno_name in test_path_list:
        anno_path = anno_prefix + anno_name
        sign_anno_reader = open(anno_path, "r")
        image_anno_list = json.load(sign_anno_reader)
        sign_anno_reader.close()
        for anno_info in image_anno_list:
            sign_name = idx_sign_dict[anno_info[-1]]
            save_path = save_template.format(sign_name)
            save_handler = open(save_path, "a")
            record_info = " ".join(
                [anno_name[:-5], "1"] + [str(pos) for pos in anno_info[:-1]])
            save_handler.write(record_info)
            save_handler.write("\n")
            save_handler.close()
# create_test_targets()

# 根据模型产生的结果，使用相应的评价标准进行评估
def evaluate_test_results():
    pass