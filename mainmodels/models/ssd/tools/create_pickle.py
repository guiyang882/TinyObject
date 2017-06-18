# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
Create raw data pickle file
data_raw is a dict mapping image_filename -> [{'class': class_int, 'box_coords': (x1, y1, x2, y2)}, {...}, ...]
'''
import json
import os
import pickle
import re

import numpy as np
from PIL import Image

from mainmodels.models.ssd.settings import g_SSDConfig


# Script config
RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
# TARGET_W, TARGET_H = 400, 260  # 1.74 is weighted avg ratio, but 1.65 aspect ratio is close enough (1.65 was for stop signs)
GRAYSCALE = True if g_SSDConfig.NUM_CHANNELS == 1 else False
TARGET_W, TARGET_H = g_SSDConfig.IMG_W, g_SSDConfig.IMG_H

###########################
# Execute main script
###########################

# First get mapping from sign name string to integer label
# sign_map = {'stop': 1,
#       'pedestrianCrossing': 2}  # only 2 sign classes (background class is 0)
sign_map = dict()
if not os.path.exists(g_SSDConfig.tt100k_traffic_sign_path):
    raise IOError("%s file not found !" % g_SSDConfig.tt100k_traffic_sign_path)
with open(g_SSDConfig.tt100k_traffic_sign_path, 'r') as handle:
    sign_map = json.load(handle)

print(sign_map)

# Create raw data pickle file
def prepare_train_raw_data():
    data_raw = {}
    if not os.path.exists(g_SSDConfig.tt100k_train_annotation_path):
        raise IOError("%s not found !" % g_SSDConfig.tt100k_train_annotation_path)
    with open(g_SSDConfig.tt100k_train_annotation_path, "r") as handle:
        train_annotation_json = json.load(handle)
        for abs_file_path, target_pos_dict in train_annotation_json.items():
            prefix_file_path = "/".join(abs_file_path.split("/")[-2:])
            box_coords_list = list()
            for label, pos_list in target_pos_dict.items():
                for cell in pos_list:
                    cell_dict = dict()
                    cell_dict["class"] = sign_map[label]
                    cell_dict["box_coords"] = np.round(np.array(cell))
                    box_coords_list.append(cell_dict)
            data_raw[prefix_file_path] = box_coords_list

    with open(g_SSDConfig.TRAIN_DATA_RAW_PATH, 'wb') as f:
        pickle.dump(data_raw, f)

def prepare_test_raw_data():
    data_raw = {}
    if not os.path.exists(g_SSDConfig.tt100k_test_annotation_path):
        raise IOError(
            "%s not found !" % g_SSDConfig.tt100k_test_annotation_path)
    with open(g_SSDConfig.tt100k_test_annotation_path, "r") as handle:
        test_annotation_json = json.load(handle)
        for abs_file_path, target_pos_dict in test_annotation_json.items():
            prefix_file_path = "/".join(abs_file_path.split("/")[-2:])
            box_coords_list = list()
            for label, pos_list in target_pos_dict.items():
                for cell in pos_list:
                    cell_dict = dict()
                    cell_dict["class"] = sign_map[label]
                    cell_dict["box_coords"] = np.round(np.array(cell))
                    box_coords_list.append(cell_dict)
            data_raw[prefix_file_path] = box_coords_list

    with open(g_SSDConfig.TEST_DATA_RAW_PATH, 'wb') as f:
        pickle.dump(data_raw, f)

prepare_test_raw_data()