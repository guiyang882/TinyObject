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

def split_data(pkl_path, split_length=5000):
    if not os.path.exists(pkl_path):
        raise IOError("%s not found !" % pkl_path)
    with open(pkl_path, "rb") as handle:
        raw_data = pickle.load(handle)
        cnt = 0
        sub_raw_data = dict()
        for key, val in raw_data.items():
            cnt += 1
            sub_raw_data[key] = val
            if cnt % split_length == 0:
                base_name = ".".join(
                    os.path.basename(pkl_path).split(".")[0:-1])
                sub_save_path = "/".join(
                    pkl_path.split("/")[0:-1] +
                    [base_name + "_" + str(cnt // split_length) + ".pkl"])
                print(sub_save_path)
                with open(sub_save_path, "wb") as save_handle:
                    pickle.dump(sub_raw_data, save_handle)
                sub_raw_data = dict()

def prepare_raw_data_with_resized(tt100k_annotation_path):
    # Create raw data pickle file
    data_raw = {}
    if not os.path.exists(tt100k_annotation_path):
        raise IOError(
            "%s not found !" % tt100k_annotation_path)

    isTestDataSets = False
    if "test" in tt100k_annotation_path:
        isTestDataSets = True

    print("Start deal with %s ..." % tt100k_annotation_path)
    with open(tt100k_annotation_path, "r") as handle:
        test_annotation_json = json.load(handle)
        for abs_file_path, target_pos_dict in test_annotation_json.items():
            image = Image.open(abs_file_path)
            orig_w, orig_h = image.size
            if g_SSDConfig.NUM_CHANNELS == 1:
                image = image.convert('L')  # 8-bit grayscale
            image = image.resize((TARGET_W, TARGET_H),
                                 Image.LANCZOS)  # high-quality downsampling filter
            if isTestDataSets:
                resized_image_save_dir = "/".join([
                    g_SSDConfig.RESIZED_IMAGES_DIR, "test",
                    "resized_images_%dx%d" % (TARGET_H, TARGET_W)])
            else:
                resized_image_save_dir = "/".join([
                    g_SSDConfig.RESIZED_IMAGES_DIR, "train",
                    "resized_images_%dx%d" % (TARGET_H, TARGET_W)])
            if not os.path.isdir(resized_image_save_dir):
                os.makedirs(resized_image_save_dir)

            resize_image_file_path = os.path.join(resized_image_save_dir,
                                                  os.path.basename(abs_file_path))
            image.save(resize_image_file_path)

            # Rescale box coordinates
            x_scale = TARGET_W / orig_w
            y_scale = TARGET_H / orig_h

            box_coords_list = list()
            for label, pos_list in target_pos_dict.items():
                for box_coords in pos_list:
                    cell_dict = dict()
                    cell_dict["class"] = sign_map[label]
                    ulc_x, ulc_y, lrc_x, lrc_y = box_coords
                    new_box_coords = (
                        ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale,
                        lrc_y * y_scale)
                    new_box_coords = [round(x) for x in new_box_coords]
                    box_coords = np.array(new_box_coords)
                    cell_dict["box_coords"] = box_coords
                    box_coords_list.append(cell_dict)
            data_raw[resize_image_file_path] = box_coords_list

    raw_save_path = g_SSDConfig.TRAIN_DATA_RAW_PATH
    if isTestDataSets:
        raw_save_path = g_SSDConfig.TEST_DATA_RAW_PATH
    with open(raw_save_path, 'wb') as f:
        pickle.dump(data_raw, f)


if __name__ == '__main__':
    # prepare_train_raw_data()
    # prepare_test_raw_data()
    prepare_raw_data_with_resized(g_SSDConfig.tt100k_train_annotation_path)
    prepare_raw_data_with_resized(g_SSDConfig.tt100k_test_annotation_path)
    split_data(g_SSDConfig.TRAIN_DATA_RAW_PATH)
    split_data(g_SSDConfig.TEST_DATA_RAW_PATH)