# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/7/8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import codecs
import random
import cv2

from mainmodels.dataset.NWPUVHR10 import nwpu_config

# 主要准备训练数据的代码和部分统计信息代码

dir_prefix = "/Volumes/projects/repos/RSI/NWPUVHR10/"
annotation_dir = dir_prefix + "sub_annotation/"
image_dir = dir_prefix + "large_src/"

target_save_dir = dir_prefix + "crop_target/"
neg_samples_save_dir = dir_prefix + "neg_sample/"
ssd_sample_dir = dir_prefix + "ssd_resalexnet/"


SSD_IMG_W, SSD_IMG_H = 512, 512
RANDOM_SAMPLE_NUM = 500  # 随机采样500张原始图像用来生成训练样本
RANDOM_SAMPLE_FROM_EACH_IMAGE = 15

sign_idx_dict = nwpu_config.sign_idx_dict
idx_sign_dict = nwpu_config.idx_sign_dict

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
def extract_target_posinfo(filename):
    if not os.path.exists(filename):
        raise IOError(filename + " not exists !")
    with codecs.open(filename, 'r', 'utf8') as handler:
        res = []
        for line in handler:
            if len(line) <= 5:
                continue
            line = line.strip().split(',')
            label_id = int(line[-1])
            tmp_pos = []
            for item in line[:-1]:
                tmp = int(item.strip('()'))
                tmp_pos.append(tmp)
            tmp_pos.append(label_id)
            res.append(tmp_pos)
    print(res)
    return res

# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples():
    def _crop_valid(area, anno_targets):
        anno_res = []
        for info in anno_targets:
            if ((info[0] >= area[0] and info[1] >= area[1]) and
                (info[2] <= area[2] and info[3] <= area[3])):
                anno_res.append([info[0] - area[0], info[1] - area[1],
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
                    anno_res.append([info[0] - area[0], info[1] - area[1],
                                     x_max_min - area[0], y_max_min - area[1],
                                     info[-1]])
        return anno_res

    def _random_crop(src_image, anno_targets, save_prefix):
        img_height, img_width = src_image.shape[:2]
        if img_height <= SSD_IMG_H or img_width <= SSD_IMG_W:
            return

        crop_list, anno_list = [], []
        map_status = [[[]] * (img_height-SSD_IMG_H+1)] * (img_width-SSD_IMG_W+1)
        def _update_map(r_x, r_y):
            for i in range(max(0, r_x-5), min(r_x+5, img_width-SSD_IMG_W)):
                for j in range(max(0, r_y-5), min(r_y+5, img_height-SSD_IMG_H)):
                    map_status[i][j] = 1

        rand_num = 0
        while len(crop_list) <= RANDOM_SAMPLE_FROM_EACH_IMAGE:
            rand_num += 1
            if rand_num > img_width*img_height*RANDOM_SAMPLE_FROM_EACH_IMAGE:
                break
            u_x = random.randint(0, img_width - SSD_IMG_W)
            u_y = random.randint(0, img_height - SSD_IMG_H)
            if map_status[u_x][u_y]:
                continue
            _update_map(u_x, u_y)
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
        if annotation_file.startswith("._"):
            continue
        abs_anno_path = annotation_dir + annotation_file
        anno_targets = extract_target_posinfo(abs_anno_path)
        abs_src_path = image_dir + ".".join(
            annotation_file.split(".")[:-1] + ["jpg"])
        if not os.path.exists(abs_src_path):
            continue
        image_name = ".".join(annotation_file.split(".")[:-1])
        src_img = cv2.imread(abs_src_path)
        _random_crop(src_image=src_img, anno_targets=anno_targets,
                     save_prefix=image_name)


if __name__ == "__main__":
    create_ssd_training_samples()