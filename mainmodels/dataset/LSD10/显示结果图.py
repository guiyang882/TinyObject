# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from mainmodels.dataset.LSD10.tools import extract_target_from_xml
from mainmodels.dataset.LSD10.lsd_config import sign_idx_dict, idx_sign_dict

dir_prefix = "/Volumes/projects/repos/RSI/LSD10/"
anno_dir = dir_prefix + "Annotations/"
image_dir = dir_prefix + "JPEGImages/"
model_res_dir = dir_prefix + "results/"
all_test_filepath = dir_prefix + "all_test.txt"

def merge_model_results(prob=0.5):
    # {"f_name": [[prob,area,target],]}
    all_model_results = dict()
    for filename in os.listdir(model_res_dir):
        target_name = filename.split(".")[0].split("_")[-1]
        model_res_path = model_res_dir + filename
        with open(model_res_path, "r") as res_reader:
            for line in res_reader:
                line = line.strip().split(" ")
                if float(line[1]) < prob:
                    continue
                f_name, t_prob = line[0]+".jpg", float(line[1])
                t_area = [int(float(a)) for a in line[2:]]
                if f_name not in all_model_results:
                    all_model_results[f_name] = list()
                all_model_results[f_name].append([t_prob]+t_area+[target_name])
    return all_model_results

all_model_results = merge_model_results()

def get_true_target_name(input_name):
    return idx_sign_dict[sign_idx_dict[input_name]]

with open(all_test_filepath, "r") as test_reader:
    for line in test_reader:
        filename = image_dir + line.strip()
        gt_anno_path = anno_dir + ".".join(
            line.strip().split(".")[:-1] + ["xml"])
        anno_details = extract_target_from_xml(gt_anno_path)
        for item in anno_details:
            item[-1] = get_true_target_name(item[-1])
        # print(anno_details)
        src_img = cv2.imread(filename)
        for area in anno_details:
            cv2.rectangle(src_img,
                          (area[0], area[1]),
                          (area[2], area[3]),
                          (0, 255, 0), 2)
        # 绘制模型检测结果的目标位置
        # print(all_model_results[line.strip()])
        for items in all_model_results[line.strip()]:
            cv2.rectangle(src_img,
                          (items[1], items[2]),
                          (items[3], items[4]),
                          (0, 0, 255), 2)
        cv2.imshow(line.strip(), src_img)
        cv2.waitKey()
