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

from mainmodels.models.ssd.settings import g_SSDConfig

IMG_H, IMG_W = g_SSDConfig.IMG_H, g_SSDConfig.IMG_W
FM_SIZES = g_SSDConfig.FM_SIZES
DEFAULT_BOXES = g_SSDConfig.DEFAULT_BOXES


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
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/"
    if not os.path.isdir(dir_path):
        raise IOError("Not Found !")
    anno_filelist = os.listdir(dir_path + "ssd_anchors")
    print(len(anno_filelist))
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
        if len(total_anno_info.keys()) >= 3000:
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

    def _visulaization(src_img, y_true_label, y_true_loc):
        gt_box_list, db_box_list = list(), list()
        base_scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        y_true_idx = 0
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        if y_true_label[y_true_idx] == 0:
                            y_true_idx += 1
                            continue
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])

                        x0_off, y0_off, x1_off, y1_off = db
                        abs_db_box_coords = np.array([
                            max(0, col+x0_off),
                            max(0, row + y0_off),
                            min(fm_w, col + 1 + x1_off),
                            min(fm_h, row + 1 + y1_off)
                        ])
                        db_box_coords = abs_db_box_coords / scale
                        gt_db_box_coords = db_box_coords * base_scale
                        print(gt_db_box_coords)
                        db_box_list.append(gt_db_box_coords)

                        abs_box_center = np.array(
                            [col+0.5, row+0.5, col+0.5, row+0.5])
                        norm_box_coords = y_true_loc[y_true_idx*4:y_true_idx*4+4]
                        abs_gt_box_coords = norm_box_coords + abs_box_center
                        gt_box_coords = abs_gt_box_coords / scale * base_scale
                        print(gt_box_coords)
                        gt_box_list.append(gt_box_coords)
                        y_true_idx += 1

        # 开始在原始图中进行目标区域的绘制 - GT
        for gt_area in map(lambda x: np.asarray(
                np.round(x), dtype=np.int32), gt_box_list):
            cv2.rectangle(src_img, (gt_area[0], gt_area[1]),
                          (gt_area[2], gt_area[3]), (0, 0, 255), 2)
        for db_area in map(lambda x: np.asarray(
            np.round(x), dtype=np.int32), db_box_list):
            cv2.rectangle(src_img, (db_area[0], db_area[1]),
                          (db_area[2], db_area[3]), (0, 255, 0), 1)
        cv2.imshow("src_ssd_gt", src_img)
        cv2.waitKey()


    dir_path = "/Volumes/projects/第三方数据下载/JL1ST" \
               "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/"
    if not os.path.isdir(dir_path):
        raise IOError("%s Not Found !" % dir_path)
    anno_filelist = os.listdir(dir_path + "ssd_anchors")
    for file in anno_filelist:
        if file.startswith("."):
            continue
        abs_anno_path = dir_path + "ssd_anchors/" + file
        image_name = "_".join(file.strip().split("_")[:-1]) + ".png"
        abs_src_image = dir_path + "src/" + image_name
        if not os.path.exists(abs_src_image):
            raise IOError("%s Not Found ! In Copy File !" % abs_src_image)
        y_true_label, y_true_loc = parse_anno_file(abs_anno_path)
        print(abs_anno_path)
        src_image = cv2.imread(abs_src_image)
        _visulaization(src_image, y_true_label, y_true_loc)
        break


if __name__ == "__main__":
    show_ssd_samples()