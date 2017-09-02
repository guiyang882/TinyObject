# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/8/24

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 将C++产生的结果编程pkl文件
import os, shutil
import pickle as pkl
import numpy as np
import cv2
import codecs
import json

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.tools.data_prep import calc_iou
from mainmodels.dataset.tools import rand_selected_file


IMG_H, IMG_W = g_SSDConfig.IMG_H, g_SSDConfig.IMG_W
FM_SIZES = g_SSDConfig.FM_SIZES
DEFAULT_BOXES = g_SSDConfig.DEFAULT_BOXES
NUM_DEFAULT_BOXES = g_SSDConfig.NUM_DEFAULT_BOXES
IOU_THRESH = g_SSDConfig.IOU_THRESH


# 根据输入的标注文件，将标注数据解析成对应的数据
def parse_anno_file(abs_anno_path):
    with open(abs_anno_path, "r") as handle:
        anno_info = handle.readlines()
        y_true_len = len(anno_info)
        y_true_label = np.zeros(y_true_len)
        y_true_loc = np.zeros(y_true_len * 4)
        for i in range(y_true_len):
            info = list(map(float, anno_info[i].strip().split(",")))
            y_true_label[i] = int(info[0])
            y_true_loc[i * 4: i * 4 + 4] = info[1:]

    return y_true_label, y_true_loc


def build_ssd_samples():
    dir_path = "/Volumes/projects/第三方数据下载/JL1ST" \
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD_AlexNet/"
    if not os.path.isdir(dir_path):
        raise IOError("Not Found !")
    anno_filelist = os.listdir(dir_path + "ssd_anchors")
    anno_filelist = rand_selected_file(anno_filelist, 1000)
    total_anno_info = dict()
    save_cnt = 0
    for file in anno_filelist:
        if file.startswith("."):
            continue
        image_name = "_".join(file.strip().split("_")[:-1]) + ".png"
        abs_anno_path = dir_path + "ssd_anchors/" + file
        print(image_name)
        with open(abs_anno_path, "r") as handle:
            anno_info = handle.readlines()
            y_true_len = len(anno_info)
            y_true_conf = np.zeros(y_true_len)
            y_true_loc = np.zeros(y_true_len * 4)
            for i in range(y_true_len):
                info = list(map(float, anno_info[i].strip().split(",")))
                y_true_conf[i] = int(info[0])
                y_true_loc[i * 4: i * 4 + 4] = info[1:]
            total_anno_info[image_name] = {
                "y_true_conf": y_true_conf,
                "y_true_loc": y_true_loc
            }
        if len(total_anno_info.keys()) >= 1000:
            save_abs_file = dir_path + "ssd_sample_prepare_%d.pkl" % save_cnt
            print(save_abs_file)
            with open(save_abs_file, 'wb') as handle:
                pkl.dump(total_anno_info, handle)
            save_cnt += 1
            total_anno_info.clear()
            print(total_anno_info.keys())
    if len(total_anno_info.keys()):
        save_abs_file = dir_path + "ssd_sample_prepare_%d.pkl" % save_cnt
        print(save_abs_file)
        with open(save_abs_file, 'wb') as handle:
            pkl.dump(total_anno_info, handle)
        save_cnt += 1
        total_anno_info.clear()


def copy_ssd_images():
    dir_path = "/Volumes/projects/第三方数据下载/JL1ST" \
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/"
    if not os.path.isdir(dir_path):
        raise IOError("Not Found !")
    anno_filelist = os.listdir(dir_path + "ssd_anchors")
    for file in anno_filelist:
        if file.startswith("."):
            continue
        image_name = "_".join(file.strip().split("_")[:-1]) + ".png"
        abs_src_image = dir_path + "src/" + image_name
        if not os.path.exists(abs_src_image):
            raise IOError("%s Not Found ! In Copy File !" % abs_src_image)
        abs_dst_path = dir_path + "ssd_src/" + image_name
        shutil.copy(abs_src_image, abs_dst_path)


def show_ssd_samples():
    # 将SSD网络中的对应的anchor box可视化出来

    def _visulaization(src_img, anno_box_pos):
        for gt_box_coords in anno_box_pos:
            cv2.rectangle(src_img,
                          (int(gt_box_coords[0]), int(gt_box_coords[1])),
                          (int(gt_box_coords[2]), int(gt_box_coords[3])),
                          (0, 255, 0), 1)
        cv2.imshow("src_ssd_gt", src_img)
        cv2.waitKey()

    def _parse_selectex_box(anno_file_path):
        if not os.path.exists(anno_file_path):
            raise IOError("%s not found ! in show_ssd_samples !" % anno_file_path)
        _anno_list = []
        with codecs.open(anno_file_path, "r", "utf8") as handler:
            for line in handler:
                line = list(map(float, line.strip().split(",")))
                line = list(map(int, line))
                _anno_list.append(line)
        return _anno_list


    dir_path = "/Volumes/projects/第三方数据下载/JL1ST" \
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD_AlexNet/"
    if not os.path.isdir(dir_path):
        raise IOError("%s Not Found !" % dir_path)
    anno_filelist = os.listdir(dir_path + "ssd_label_box_src")
    for file in anno_filelist:
        if file.startswith("."):
            continue
        abs_anno_path = dir_path + "ssd_label_box_src/" + file
        image_name = "_".join(file.strip().split("_")[:-1]) + ".png"
        abs_src_image = dir_path + "src/" + image_name
        if not os.path.exists(abs_src_image):
            # raise IOError("%s Not Found ! In Copy File !" % abs_src_image)
            print("%s Not Found ! In Copy File !" % abs_src_image)
            continue
            # raise IOError("%s Not Found ! In Copy File !" % abs_src_image)
        src_image = cv2.imread(abs_src_image)
        anno_box_pos = _parse_selectex_box(abs_anno_path)
        _visulaization(src_image, anno_box_pos)
        # break


def _find_gt_boxes(src_image, signs_data):
    """
    Given (global) feature map sizes, and single training example,
    find all default boxes that exceed Jaccard overlap threshold

    Returns y_true array that flags the matching default boxes with class ID (-1 means nothing there)
    """
    # Pre-process ground-truth data
    # Convert absolute coordinates to relative coordinates ranging from 0 to 1
    # Read the sign class label (note background class label is 0, sign labels are ints >=1)

    signs_box_coords = []  # relative coordinates
    for abs_box_coords in signs_data:
        # Calculate relative coordinates
        # (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner
        scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        box_coords = np.array(abs_box_coords) / scale
        signs_box_coords.append(box_coords)

    # For each GT box, for each feature map, for each feature map cell, for each default box:
    # 1) Calculate the Jaccard overlap (IOU) and annotate the class label
    # 2) Count how many box matches we got
    # 3) If we got a match, calculate normalized box coordinates and updte y_true_loc
    for gt_box_coords in signs_box_coords:
        # for fm_idx, fm_size in enumerate(FM_SIZES):
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size  # feature map height and width
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Calculate relative box coordinates for this default box
                        x1_offset, y1_offset, x2_offset, y2_offset = db
                        abs_db_box_coords = np.array([
                            max(0, col + x1_offset),
                            max(0, row + y1_offset),
                            min(fm_w, col + 1 + x2_offset),
                            min(fm_h, row + 1 + y2_offset)
                        ])
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])
                        db_box_coords = abs_db_box_coords / scale

                        # Calculate Jaccard overlap (i.e. Intersection Over Union, IOU) of GT box and default box
                        iou = calc_iou(gt_box_coords, db_box_coords)

                        # If box matches, i.e. IOU threshold met
                        if iou >= IOU_THRESH:
                            # Calculate normalized box coordinates and update y_true_loc
                            # absolute coordinates of center of feature map cell
                            abs_box_center = np.array(
                                [col + 0.5, row + 0.5])
                            # absolute ground truth box coordinates (in feature map grid)
                            abs_gt_box_coords = gt_box_coords * scale
                            norm_box_coords = abs_gt_box_coords - np.concatenate(
                                (abs_box_center, abs_box_center))
                            print(norm_box_coords)


# 分析为何对于给定的样本，没有SSD的anchor的label
def analysis_ssd_sample_no_label():
    dir_path = "/Volumes/projects/第三方数据下载/JL1ST" \
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/"
    src_img_list = dir_path + "no_ssd_anno.txt"
    with codecs.open(src_img_list, "r", "utf8") as handle:
        for line in handle:
            abs_img_file_name = line.strip()
            src_img = cv2.imread(abs_img_file_name)
            anno_file = ".".join(abs_img_file_name.split("/")[-1].split(".")[
                        :-1]) + ".json"
            with codecs.open(dir_path + "annotation/" + anno_file, "r") as handle:
                anno_targets = json.load(handle)
                for area in anno_targets:
                    print(area)
                    cv2.rectangle(src_img,
                                  (area[0], area[1]), (area[2], area[3]),
                                  (0, 0, 255), 2)
            _find_gt_boxes(src_img, np.asarray(anno_targets))
            # cv2.imshow("src", src_img)
            # cv2.waitKey()
            break


# 将不同批次的ssd样本的结果进行整合
def combine_ssd_samples(datatype="train"):
    dir_path = "/home/ai-i-liuguiyang/repos_ssd/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/ssd_prepare"
    file_list = ["ssd_sample_prepare_%d.pkl" % i for i in range(0, 3)]
    res = dict()
    for item in file_list:
        abs_file_path = dir_path + "/" + item
        with open(abs_file_path, "rb") as handler:
            tmp = pkl.load(handler)
            res.update(tmp)
    if datatype == "train":
        fname = "train_data_prep.pkl"
    else:
        fname = "test_data_prep.pkl"
    with open(dir_path + "/" + fname, "wb") as handler:
        pkl.dumps(res, handler)


if __name__ == "__main__":
    build_ssd_samples()