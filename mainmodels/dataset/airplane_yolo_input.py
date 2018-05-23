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
import xml.dom.minidom
import cv2

from mainmodels.models.ssd.settings import g_SSDConfig

# 解析XML文本得到JL1ST的目标数据
dir_prefix = "/Volumes/projects/第三方数据下载/JL1ST/"
JL1ST_NAME = "JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS"
annotation_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_annotation/"
image_dir = dir_prefix + "SRC_" + JL1ST_NAME + "/"
ssd_sample_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_YOLOv2/"
# SSD_IMG_W, SSD_IMG_H = g_SSDConfig.IMG_W, g_SSDConfig.IMG_H
SSD_IMG_W, SSD_IMG_H = 608, 608
RANDOM_SAMPLE_NUM = 500  # 随机采样500张原始图像用来生成训练样本


# 采用蓄水池采样算法对序列进行采样
def rand_selected_file(file_list):
    res = list()
    for i in range(0, len(file_list)):
        if i < RANDOM_SAMPLE_NUM:
            res.append(file_list[i])
        else:
            M = random.randint(0, i)
            if M < RANDOM_SAMPLE_NUM:
                res[M] = file_list[i]
    return res

# 给定一个标记文件，找到对应的目标的位置信息
def extract_airplane_posinfo(filename):
    if not os.path.exists(filename):
        raise IOError(filename + " not exists !")
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    # 获取集合中所有的目标
    targets = collection.getElementsByTagName("object")
    res = []
    for target in targets:
        target_name = target.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = target.getElementsByTagName("bndbox")[0]
        xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data
        res.append([int(xmin), int(ymin), int(xmax), int(ymax), target_name])
    return res


# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples():
    def _crop_valid(area, anno_targets):
        anno_res = []
        for info in anno_targets:
            if ((info[0] >= area[0] and info[1] >= area[1]) and
                (info[2] <= area[2] and info[3] <= area[3])):
                anno_res.append([info[0] - area[0], info[1] - area[1],
                                 info[2] - area[0], info[3] - area[1]])
            if (info[0] >= area[0] and info[1] >= area[1] and
                info[0] < area[2] and info[1] < area[3] and
                (not (info[2] <= area[2] and info[3] <= area[3]))):
                base = (info[2] - info[0]) * (info[3] - info[1])
                x_max_min = min(info[2], area[2])
                y_max_min = min(info[3], area[3])
                new_square = (x_max_min - info[0]) * (y_max_min - info[1])
                if new_square / base >= 0.75:
                    anno_res.append([info[0] - area[0], info[1] - area[1],
                                     x_max_min - area[0], y_max_min - area[1]])
        return anno_res

    def _random_crop(src_image, anno_targets, save_prefix):
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
        # 保存裁剪出来的结果
        if not os.path.isdir(ssd_sample_dir):
            os.makedirs(ssd_sample_dir)
        if not os.path.isdir(ssd_sample_dir + "src"):
            os.makedirs(ssd_sample_dir + "src")
        if not os.path.isdir(ssd_sample_dir + "annotation"):
            os.makedirs(ssd_sample_dir + "annotation")

        for i in range(len(crop_list)):
            x0, y0, x1, y1 = crop_list[i]
            # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
            sub_img = src_img[y0:y1, x0:x1]
            f_name = save_prefix + "_%d_%d_%d_%d_%d.png" % (x0, y0, x1, y1, i)
            cv2.imwrite(ssd_sample_dir + "src/" + f_name, sub_img)
            f_name = save_prefix + "_%d_%d_%d_%d_%d.json" % (x0, y0, x1, y1, i)
            with codecs.open(
                        ssd_sample_dir + "annotation/" + f_name, "w") as handle:
                handle.write(json.dumps(anno_list[i]))
            print("save %s" % f_name)


    if not (os.path.exists(annotation_dir) and os.path.isdir(annotation_dir)):
        raise IOError("%s Not Found !" % annotation_dir)
    annotation_lists = os.listdir(annotation_dir)

    annotation_lists = rand_selected_file(annotation_lists)
    for annotation_file in annotation_lists:
        abs_anno_path = annotation_dir + annotation_file
        anno_targets = extract_airplane_posinfo(abs_anno_path)
        abs_src_path = image_dir + ".".join(
            annotation_file.split(".")[:-1] + ["png"])
        if not os.path.exists(abs_src_path):
            continue
        image_name = ".".join(annotation_file.split(".")[:-1])
        src_img = cv2.imread(abs_src_path)
        _random_crop(src_image=src_img, anno_targets=anno_targets,
                     save_prefix=image_name)
        # break


if __name__ == "__main__":
    create_ssd_training_samples()