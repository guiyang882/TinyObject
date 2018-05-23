# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/9

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from mainmodels.dataset import tools

score_csuvideo_path = "/Volumes/projects/repos/RSI/CSUVOCFormat/data/CSUVIDEO" \
                      "/results/LDSSD_512x512_score/Main/comp4_det_test_plane.txt"
test_img_csuvideo_dir = "/Volumes/projects/repos/RSI/CSUVOCFormat/512x512" \
                        "/large_tunisia_total/JPEGImages/"
test_anno_csuvideo_dir = "/Volumes/projects/repos/RSI/CSUVOCFormat/512x512" \
                         "/large_tunisia_total/Annotations/"


# 将标准的标签数据转换成和VOC一致的txt数据格式进行对比
def convert_std_annotation(video_name):
    data_dir = "/Volumes/projects/repos/RSI/CSUVOCFormat/"
    tpl_save_dir = "/Volumes/projects/repos/RSI/CSUVOCFormat/data/CSUVIDEO" \
                   "/results/LDSSD_512x512_score/Main/comp4_det_std_{" \
                   "}_plane.txt"
    tpl_anno_dir = data_dir + "{}x{}/{}/Annotations/"
    anno_dir = tpl_anno_dir.format(512, 512, video_name)
    std_writer = open(tpl_save_dir.format(video_name), "w")
    for xml_name in os.listdir(anno_dir):
        abs_anno_path = anno_dir + xml_name
        anno_targets = tools.extract_airplane_posinfo(abs_anno_path)
        for area in anno_targets:
            std_writer.write("{} 1 {} {} {} {}\n".format(
                ".".join(xml_name.split(".")[:-1]),
                area[0], area[1], area[2], area[3]))
    std_writer.close()


def show_res():
    with open(score_csuvideo_path, "r") as score_reader:
        imgs_info = dict()
        for line in score_reader:
            line = line.strip().split(" ")
            img_name, prob, xmin, ymin, xmax, ymax = line[:]
            if img_name not in imgs_info:
                imgs_info[img_name] = list()
            imgs_info[img_name].append([float(prob), int(xmin), int(ymin),
                                        int(xmax), int(ymax)])
        print(len(imgs_info))
        for img_name, target_annos in imgs_info.items():
            abs_img_path = test_img_csuvideo_dir + img_name + ".jpg"
            print(abs_img_path)
            if not os.path.exists(abs_img_path):
                print(abs_img_path)
                continue

            abs_anno_path = test_anno_csuvideo_dir + img_name + ".xml"
            gt_list = tools.extract_airplane_posinfo(abs_anno_path)
            src_img = cv2.imread(abs_img_path)
            w, h = src_img.shape[:2]
            if w != 512 or h != 512:
                continue
            for prob_area in target_annos:
                prob = prob_area[0]
                # print(prob)
                if prob < 0.40:
                    continue
                area = prob_area[1:]
                cv2.rectangle(src_img, (area[0], area[1]), (area[2], area[3]),
                              (0, 255, 0), 2)
            for area in gt_list:
                cv2.rectangle(src_img, (area[0], area[1]), (area[2], area[3]),
                              (0, 0, 255), 2)
            cv2.imshow("src_img", src_img)
            cv2.waitKey()


# convert_std_annotation("large_000013363_total")
# convert_std_annotation("large_000014631_total")
# convert_std_annotation("large_minneapolis_1_total")
# convert_std_annotation("large_tunisia_total")
show_res()
