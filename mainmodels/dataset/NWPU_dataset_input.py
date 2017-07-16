# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/7/8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pickle as pkl
import os

import cv2
import matplotlib.pyplot as plt

from mainmodels.dataset.tools import show_image_with_annotation
from mainmodels.dataset.tools import UtilityTools
from mainmodels.dataset.tools import Point
from mainmodels.dataset.tools import SampleStep
from mainmodels.dataset.tools import Rectangle
from mainmodels.dataset.tools import BBox


candidate_labels = ["rotation"+str(i*45) for i in range(1, 8)]

class NWPU_DataSet():

    def init(self):
        base_dir = "/Volumes/projects/NWPU-VHR-10-dataset"
        sub_path_all = os.listdir(base_dir)
        label_json_path_all = []
        raw_label_path = ""
        for sub_path in sub_path_all:
            path = os.path.join(base_dir, sub_path)
            if sub_path == "src":
                raw_label_path = path + os.sep + "label.pkl"
                continue
            target_label_path = path + os.sep + "label.pkl"
            if os.path.exists(target_label_path):
                for candidate_name in candidate_labels:
                    if candidate_name in target_label_path:
                        label_json_path_all.append(target_label_path)
        if raw_label_path:
            label_json_path_all.append(raw_label_path)

        total_image_dict = dict()
        for lable_cell in label_json_path_all:
            if not os.path.exists(lable_cell):
                raise IOError("label json path %s not exists !" % lable_cell)
            with open(lable_cell, "rb") as handle:
                all_label = pkl.load(handle)
                for file_path, label_info in all_label.items():
                    new_abs_file_path = os.path.join(base_dir, os.sep.join(
                        list(file_path.split('\\')[2:])))
                    total_image_dict[new_abs_file_path] = label_info
                    # print(new_abs_file_path)
                    # self.show_label_info(new_abs_file_path, label_info)
        with open(base_dir + "/" + "train_annotation.pkl", "wb") as handle:
            pkl.dump(total_image_dict, handle)
        print("NWPU train_annotation save !")

    def show_label_info(self, img_file_path, img_label_list):
        if not os.path.exists(img_file_path):
            raise IOError("image file path %s not Exists !" % img_file_path)
        img = cv2.imread(img_file_path)
        for each_file_label in img_label_list:
            (l_x, l_y, r_x, r_y) = each_file_label['box_coords']
            cv2.rectangle(img, (math.floor(l_x), math.floor(l_y)),
                          (math.ceil(r_x), math.ceil(r_y)), (55, 255, 155), 4)

        plt.imshow(img)