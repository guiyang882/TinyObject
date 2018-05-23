# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 在准备好原始的大图像之后，我们需要适应模型的训练，需要将大图像切割成小图像

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


SSD_IMG_W, SSD_IMG_H = 512, 512
RANDOM_SAMPLE_NUM = 500  # 随机采样500张原始图像用来生成训练样本

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

    total_anno_names = list()
    for anno_name in os.listdir(src_annotation_dir):
        if anno_name.startswith("._"):
            continue
        total_anno_names.append(anno_name)
    print(len(total_anno_names))
    sub_anno_names = tools.rand_selected_file(total_anno_names,
                                              len(total_anno_names) // 15)
    for anno_name in sub_anno_names:
        anno_path = src_annotation_dir + anno_name
        annotation_info = tools.extract_airplane_posinfo(anno_path)
        src_image = cv2.imread(src_images_dir + anno_name.replace("xml", "jpg"))
        create_ssd_training_samples(video_name, src_image, annotation_info,
                                    anno_name.replace("xml", "jpg"))


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
        for lx in range(0, w, int(SSD_IMG_W//2.5)):
            for ly in range(0, h, int(SSD_IMG_H//2.5)):
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

    def _random_crop_for_target_with_condition():
        crop_list, anno_list = [], []
        random.seed(time.asctime())
        for idx in range(0, len(anno_targets)):
            c_x = (anno_targets[idx][0] + anno_targets[idx][2]) // 2
            c_y = (anno_targets[idx][1] + anno_targets[idx][3]) // 2
            u_x = random.randint(max(0, c_x - SSD_IMG_W // 2), anno_targets[idx][0])
            u_y = random.randint(max(0, c_y - SSD_IMG_H // 2), anno_targets[idx][1])

            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
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

# 统计大图像的RGB各个通道的均值

import numpy as np
def calc_rgb_mean():
    r_list, g_list, b_list = list(), list(), list()
    with open("/Volumes/projects/repos/RSI/CSUVOCFormat/source/total.txt",
              "r") as reader:
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
'''
CSURSI-Video
118.228683948
118.142248499
119.506718973
'''

'''
TT100K
117
120
117
'''

# 检查切出来数据的大小是不是符合尺寸
def check_samples_shape(video_name):
    # tpl_img_voc_dir = dir_prefix + "{}x{}/{}/JPEGImages/"
    # tpl_annotation_voc_dir = dir_prefix + "{}x{}/{}/Annotations/"
    img_voc_dir = tpl_img_voc_dir.format(SSD_IMG_W, SSD_IMG_H, video_name)
    for img_name in os.listdir(img_voc_dir):
        abs_img_path = img_voc_dir + img_name
        src_img = cv2.imread(abs_img_path)
        w, h = src_img.shape[:2]
        if w != 512 or h != 512:
            print(video_name, img_name, w, h)

# 产生每个文件的对应的列表文件
def create_list_file(video_name):
    list_writer = open("/Volumes/projects/repos/RSI/CSUVOCFormat/512x512"
                       "/"+video_name+".txt", "w")
    img_voc_dir = tpl_img_voc_dir.format(SSD_IMG_W, SSD_IMG_H, video_name)
    tpl_prefix_img_path = "{}x{}/{}/JPEGImages/{}"
    tpl_prefix_anno_path = "{}x{}/{}/Annotations/{}"
    for img_name in os.listdir(img_voc_dir):
        anno_name = img_name.replace("jpg", "xml")
        img_path = tpl_prefix_img_path.format(
            SSD_IMG_H, SSD_IMG_W, video_name, img_name)
        anno_path = tpl_prefix_anno_path.format(
            SSD_IMG_H, SSD_IMG_W, video_name, anno_name)
        list_writer.write("{} {}\n".format(img_path, anno_path))
    list_writer.close()


# 将jpg图像转化为png格式
def convert_jpg2png(video_name):
    img_voc_dir = tpl_img_voc_dir.format(SSD_IMG_W, SSD_IMG_H, video_name)
    src_png_dir = tpl_src_png_dir.format(SSD_IMG_W, SSD_IMG_H, video_name)
    if not os.path.exists(src_png_dir):
        os.makedirs(src_png_dir)
    for img_name in os.listdir(img_voc_dir):
        png_name = img_name.replace("jpg", "png")
        abs_jpg_path = img_voc_dir + img_name
        abs_png_path = src_png_dir + png_name
        src_img = cv2.imread(abs_jpg_path)
        cv2.imwrite(abs_png_path, src_img)




if __name__ == "__main__":
    # load_annotation("large_000013363_total")
    # load_annotation("large_000014631_total")
    # load_annotation("large_minneapolis_1_total")
    # load_annotation("large_tunisia_total")
    # calc_rgb_mean()
    # check_samples_shape("large_minneapolis_1_total")
    # create_list_file("large_000013363_total")
    # create_list_file("large_000014631_total")
    # create_list_file("large_minneapolis_1_total")
    # create_list_file("large_tunisia_total")

    convert_jpg2png("large_000013363_total")
    convert_jpg2png("large_000014631_total")
    convert_jpg2png("large_minneapolis_1_total")
    convert_jpg2png("large_tunisia_total")
    pass