# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.tools.data_prep import calc_iou

def nms(y_pred_conf, y_pred_loc, prob):
    """
	Non-Maximum Suppression (NMS)
	Performs NMS on all boxes of each class where predicted probability > CONF_THRES
	For all boxes exceeding IOU threshold, select the box with highest confidence
	Returns a lsit of box coordinates post-NMS

	Arguments:
		* y_pred_conf: Class predictions, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)
		* y_pred_loc: Bounding box coordinates, numpy array of shape (num_feature_map_cells * num_defaul_boxes * 4,)
			These coordinates are normalized coordinates relative to center of feature map cell
		* prob: Class probabilities, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)

	Returns:
		* boxes: Numpy array of boxes, with shape (num_boxes, 6). shape[0] is interpreted as:
			[x1, y1, x2, y2, class, probability], where x1/y1/x2/y2 are the coordinates of the
			upper-left and lower-right corners. Box coordinates assume the image size is IMG_W x IMG_H.
			Remember to rescale box coordinates if your target image has different dimensions.
	"""
    # Keep track of boxes for each class
    class_boxes = {}  # class -> [(x1, y1, x2, y2, prob), (...), ...]
    with open('signnames.csv', 'r') as f:
        for line in f:
            cls, _ = line.split(',')
            class_boxes[float(cls)] = []

    # Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
    y_idx = 0
    for fm_size in g_SSDConfig.FM_SIZES:
        fm_h, fm_w = fm_size  # feature map height and width
        for row in range(fm_h):
            for col in range(fm_w):
                for db in g_SSDConfig.DEFAULT_BOXES:
                    # Only perform calculations if class confidence > CONF_THRESH and not background class
                    if prob[y_idx] > g_SSDConfig.CONF_THRESH and y_pred_conf[
                        y_idx] > 0.:
                        # Calculate absolute coordinates of predicted bounding box
                        xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
                        center_coords = np.array([xc, yc, xc, yc])
                        abs_box_coords = center_coords + y_pred_loc[
                                                         y_idx * 4: y_idx * 4 + 4]  # predictions are offsets to center of fm cell

                        # Calculate predicted box coordinates in actual image
                        scale = np.array([g_SSDConfig.IMG_W / fm_w,
                                          g_SSDConfig.IMG_H / fm_h,
                                          g_SSDConfig.IMG_W / fm_w,
                                          g_SSDConfig.IMG_H / fm_h])
                        box_coords = abs_box_coords * scale
                        box_coords = [int(round(x)) for x in box_coords]

                        # Compare this box to all previous boxes of this class
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

    # Gather all the pruned boxes and return them
    boxes = []
    for cls in class_boxes.keys():
        for class_box in class_boxes[cls]:
            boxes.append(class_box)
    boxes = np.array(boxes)

    return boxes
