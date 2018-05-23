# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import cv2

from mainmodels.dataset.LSD10 import tools
from mainmodels.dataset.LSD10 import lsd_config as config


nwpu_voc_dir = "/Volumes/projects/repos/RSI/NWPUVHR10/NWPUVOCFORMAT/"
nwpu_voc_image_dir = nwpu_voc_dir + "JPEGImages/"
nwpu_voc_anno_dir = nwpu_voc_dir + "Annotations/"

vedia_voc_dir = "/Volumes/projects/repos/RSI/VEDAI/VEDIAVOCFORAMT/"
vedia_voc_image_dir = vedia_voc_dir + "JPEGImages/"
vedia_voc_anno_dir = vedia_voc_dir + "Annotations/"

lsd_voc_dir = "/Volumes/projects/repos/RSI/LSD10/"
lsd_voc_image_dir = lsd_voc_dir + "JPEGImages/"
lsd_voc_anno_dir = lsd_voc_dir + "Annotations/"

# 采用蓄水池采样算法对序列进行采样
def rand_selected_file(file_list, K_ratio=2/7):
    K = int(len(file_list) * K_ratio)
    res = list()
    for i in range(0, len(file_list)):
        if i < K:
            res.append(file_list[i])
        else:
            M = random.randint(0, i)
            if M < K:
                res[M] = file_list[i]
    return res

# 先确定每个原始数据集中的训练集和测试集
def split_dataset():
    nwpu_img_list = os.listdir(nwpu_voc_image_dir)
    vedia_img_list = os.listdir(vedia_voc_image_dir)
    test_nwpu_img_list = rand_selected_file(nwpu_img_list)
    test_vedia_img_list = rand_selected_file(vedia_img_list)
    with open(nwpu_voc_dir+"test.txt", "w") as test_nwpu_writer:
        for item in test_nwpu_img_list:
            test_nwpu_writer.write("{}\n".format(item))
    with open(nwpu_voc_dir+"train.txt", "w") as train_nwpu_writer:
        for item in nwpu_img_list:
            if item not in test_nwpu_img_list:
                train_nwpu_writer.write("{}\n".format(item))
    with open(vedia_voc_dir+"test.txt", "w") as test_vedia_writer:
        for item in test_vedia_img_list:
            test_vedia_writer.write("{}\n".format(item))
    with open(vedia_voc_dir+"train.txt", "w") as train_vedia_writer:
        for item in vedia_img_list:
            if item not in test_vedia_img_list:
                train_vedia_writer.write("{}\n".format(item))


# 更新数据集中的label信息
def flush_dataset():
    for anno_name in os.listdir(lsd_voc_anno_dir):
        abs_anno_path = lsd_voc_anno_dir + anno_name
        print(abs_anno_path)
        anno_targets = tools.extract_target_from_xml(abs_anno_path)
        new_anno_targets = list()
        for anno_info in anno_targets:
            label_name = anno_info[-1]
            label_id = config.sign_idx_dict[label_name]
            label_name = config.idx_sign_dict[label_id]
            new_anno_info = anno_info[:-1] + [label_name]
            new_anno_targets.append(new_anno_info)
        src_image = cv2.imread(
            lsd_voc_image_dir+anno_name.replace("xml", "jpg"))
        xml_obj = tools.fetch_xml_format(
            src_image, anno_name.replace("xml", "jpg"), new_anno_targets)
        with open(lsd_voc_anno_dir+anno_name, "w") as writer:
            writer.write(xml_obj)

#1,$s/vehiclecar/vehicle/g

import numpy as np
def calc_rgb_mean():
    r_list, g_list, b_list = list(), list(), list()
    with open("/Volumes/projects/repos/RSI/LSD10/total.txt", "r") as reader:
        for line in reader.readlines():
            line = line.strip()
            src_img = cv2.imread(line)
            b, g, r = cv2.split(src_img)
            b_list.append(np.mean(b))
            g_list.append(np.mean(g))
            r_list.append(np.mean(r))
    print(np.mean(r_list))
    print(np.mean(g_list))
    print(np.mean(b_list))
"""
104.480289006
107.307103097
95.8043901467
"""

if __name__ == '__main__':
    # split_dataset()
    # calc_rgb_mean()
    flush_dataset()
    pass