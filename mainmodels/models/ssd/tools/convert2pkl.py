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
    pass


if __name__ == "__main__":
    build_ssd_samples()