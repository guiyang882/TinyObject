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

import numpy as np

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.tools.data_prep import calc_iou

IMG_W, IMG_H = g_SSDConfig.IMG_W, g_SSDConfig.IMG_H

def check_box_legal(box_coords, src_rect=[IMG_W, IMG_H, IMG_W, IMG_H]):
    for i in range(0, len(box_coords)):
        if box_coords[i] < 0:
            return False
        if box_coords[i] > src_rect[i]:
            return False
    return True

def nms(y_pred_conf, y_pred_loc, prob):
    print("len of prob is ", len(prob))
    print("max of prob is ", np.max(prob), "min of prob is ", np.min(prob))
    print("mean of prob is ", np.mean(prob))
    class_boxes = {}
    with open(g_SSDConfig.label_sign_path, "r") as handle:
        traffic_label_dict = json.load(handle)
        for key, val in traffic_label_dict.items():
            class_boxes[val] = list()

    class_boxes[0] = list()
    y_idx = 0
    for fm_size in g_SSDConfig.FM_SIZES:
        fm_h, fm_w = fm_size  # feature map height and width
        for row in range(fm_h):
            for col in range(fm_w):
                for db in g_SSDConfig.DEFAULT_BOXES:
                    if prob[y_idx] > g_SSDConfig.CONF_THRESH and y_pred_conf[y_idx] > 0.:
                        xc, yc = row+0.5, col+0.5
                        center_coords = np.array([xc, yc, xc, yc])
                        # predictions are offsets to center of fm cell
                        abs_box_coords = center_coords + y_pred_loc[y_idx * 4: y_idx * 4 + 4]
                        # Calculate predicted box coordinates in actual image
                        scale = np.array(
                            [IMG_W / fm_w, IMG_H / fm_h, IMG_W / fm_w,
                             IMG_H / fm_h])
                        box_coords = abs_box_coords * scale
                        box_coords = [int(round(x)) for x in box_coords]
                        if check_box_legal(box_coords):
                            cls = y_pred_conf[y_idx]
                            cls_prob = prob[y_idx]
                            box = (*box_coords, cls, cls_prob)
                            if len(class_boxes[cls]) == 0:
                                class_boxes[cls].append(box)
                            else:
                                suppressed = False  # did this box suppress other box(es)?
                                overlapped = False  # did this box overlap with other box(es)?
                                for other_box in class_boxes[cls]:
                                    iou = calc_iou(box[:4], other_box[:4])
                                    if iou > g_SSDConfig.NMS_IOU_THRESH:
                                        overlapped = True
                                        # If current box has higher confidence than other box
                                        if box[5] > other_box[5]:
                                            class_boxes[cls].remove(other_box)
                                            suppressed = True
                                if suppressed or not overlapped:
                                    class_boxes[cls].append(box)

                    y_idx += 1
    print("y_idx is ", y_idx)
    # Gather all the pruned boxes and return them
    boxes = []
    for cls in class_boxes.keys():
        for class_box in class_boxes[cls]:
            boxes.append(class_box)
    boxes = np.array(boxes)

    return boxes
