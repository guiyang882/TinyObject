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

dir_prefix = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/VOCFORMAT/ImageSets/Main/"
model_res_prefix = dir_prefix + "model_test/"
expect_res_prefix = dir_prefix + "std_test/"
sign_label_path = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/sub_traffic_sign.json"
total_test_image_path = dir_prefix + "test.txt"
model_template = "comp4_det_test_{}.txt"
expect_template = "comp4_det_std_{}.txt"

all_test_samples = set()
with open(total_test_image_path, "r") as test_sample_reader:
    for line in test_sample_reader:
        line = line.strip()
        all_test_samples.add(line)


def fetch_sign_info():
    idx_sign_dict = dict()
    with open(sign_label_path, "r") as handler:
        sign_idx_dict = json.load(handler)
        for key, val in sign_idx_dict.items():
            idx_sign_dict[val] = key
    return sign_idx_dict, idx_sign_dict


def rect_cross(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect


def rect_area(rect):
    return float(max(0.0, (rect[2] - rect[0]) * (rect[3] - rect[1])))


def calc_cover(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    return rect_area(crect) / rect_area(rect2)


def calc_iou(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    ac = rect_area(crect)
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    return ac / (a1 + a2 - ac)


# 计算各个纬度的统计数据
# False negative; Result should have been positive, but is negative.
# False positive; Result should have been negative, but is positive.
# True positive; Result should have been positive and is positive.
# True negative; Result should have been negative and is negative.
def calc_pntf_value(expect_res_dict, model_res_dict, IOU):
    """
    :param expect_res_dict: 标准数据集的GT的所有结果
    :param model_res_dict: 模型检测出来的所有结果
    :param IOU: 用来衡量候选框是不是真正的结果
    """
    tp_cnt, fn_cnt, fp_cnt = 0, 0, 0
    for image_name in model_res_dict.keys():
        if image_name in expect_res_dict.keys():
            # 是否需要考虑已经匹配过的框的位置信息?
            detected_box_idx = list()
            for i in range(len(expect_res_dict[image_name])):
                gt_box = expect_res_dict[image_name][i]["pos"]
                max_iou, max_idx, is_matched = 0.0, -1, False
                for j in range(len(model_res_dict[image_name])):
                    if j not in detected_box_idx:
                        md_box = model_res_dict[image_name][j]["pos"]
                        md_iou = calc_iou(md_box, gt_box)
                        if md_iou >= IOU:
                            # print(md_iou)
                            is_matched = True
                            max_iou = max(max_iou, md_iou)
                            if max_iou == md_iou:
                                max_idx = j
                if is_matched:
                    detected_box_idx.append(max_idx)
                    tp_cnt += 1
                else:
                    fn_cnt += 1
            fp_cnt += len(model_res_dict[image_name]) - len(detected_box_idx)
    # print("tp: {}, fp: {}, fn: {}".format(
    #     tp_cnt, fp_cnt, fn_cnt))
    # print("Precision=TP/(TP+FP)={}, Recall=TP/(TP+FN)={}".format(
    #     0 if tp_cnt == 0 else tp_cnt/(tp_cnt+fp_cnt),
    #     0 if tp_cnt == 0 else tp_cnt/(tp_cnt+fn_cnt)))
    precision = 0 if tp_cnt == 0 else tp_cnt/(tp_cnt+fp_cnt)
    recall = 0 if tp_cnt == 0 else tp_cnt/(tp_cnt+fn_cnt)
    return precision, recall



# 根据模型产生的结果，使用相应的评价标准进行评估
def evaluate_test_results(sign_name="ps", prob_limit=0.6, iou_limit=0.4):
    """
    :param sign_name: 交通标志的名字
    :param prob_limit: 给出的识别概率的下限
    :param iou_limit: GT和Box之间的IOU的下限
    """
    model_sign_name = model_template.format(sign_name)
    expect_sign_name = expect_template.format(sign_name)
    model_res_path = model_res_prefix + model_sign_name
    expect_res_path = expect_res_prefix + expect_sign_name
    # 开始组织模型的测试结果和GT结果
    model_res_dict = dict()
    with open(model_res_path, "r") as model_reader:
        for line in model_reader:
            line = line.strip().split(" ")
            prob = float(line[1])
            if prob < prob_limit:
                continue
            if line[0] not in model_res_dict.keys():
                model_res_dict[line[0]] = list()
            tmp_dict = {
                "prob": prob,
                "pos": [int(a) for a in line[2:]]
            }
            model_res_dict[line[0]].append(tmp_dict)
    expect_res_dict = dict()
    with open(expect_res_path, "r") as expect_reader:
        for line in expect_reader:
            line = line.strip().split(" ")
            if line[0] not in expect_res_dict.keys():
                expect_res_dict[line[0]] = list()
            tmp_dict = {
                "prob": 1.0,
                "pos": [int(a) for a in line[2:]]
            }
            expect_res_dict[line[0]].append(tmp_dict)
    precision, recall = calc_pntf_value(
        expect_res_dict, model_res_dict, iou_limit)
    return precision, recall

sign_idx_dict, idx_sign_dict = fetch_sign_info()
detail_dict = dict()
for sign_name in sign_idx_dict.keys():
    detail_dict[sign_name] = list()
    for iou in np.linspace(0, 1, 5):
        for prob in np.linspace(0, 1, 9):
            precision, recall = evaluate_test_results(
                sign_name=sign_name, prob_limit=prob, iou_limit=iou)
            detail_dict[sign_name].append([iou, prob, precision, recall])
print(detail_dict)
