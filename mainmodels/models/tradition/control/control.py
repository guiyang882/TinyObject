# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import cv2
import numpy as np

from mainmodels.models.tradition.control.model_cnn_server import ModelServer

proj_root = "/".join(os.path.abspath(__file__).split("/")[0:-5])
images_path = "/".join([proj_root, "trainmodel", "data", "JL1ST", "image.list"])

def prepare_sliding_windows(image, min_win_size, strid_size):
    """准备滑动窗口数据"""
    image_height, image_width = image.shape[0], image.shape[1]
    w_height, w_width = min_win_size[0], min_win_size[1]
    s_height, s_width = strid_size[0], strid_size[1]
    h_seq = [i*s_height for i in range(image_height // s_height + 1)]
    w_seq = [i*s_width for i in range(image_width // s_width + 1)]

    candidate_windows, anchorlist = [], []
    for h_st in h_seq:
        for w_st in w_seq:
            h_ed = h_st + w_height
            w_ed = w_st + w_width
            if h_st >= image_height or w_st >= image_width:
                continue
            if min(image_height, h_ed) - h_st < w_height // 2:
                continue
            if min(image_width, w_ed) - w_st < w_width // 2:
                continue
            sub_image = image[h_st:h_ed, w_st:w_ed]
            anchor = [h_st, w_st, h_ed-h_st+1, w_ed-w_st+1]
            if sub_image.shape[0] != 48 or sub_image.shape[1] != 48:
                sub_image = cv2.resize(sub_image, (48, 48),
                                       interpolation=cv2.INTER_LINEAR)
            sub_image = sub_image.reshape((48, 48, 1))
            sub_image.astype(np.float32)
            candidate_windows.append(sub_image)
            anchorlist.append(anchor)
    print(len(candidate_windows))
    return candidate_windows, anchorlist

def large2smalls_images(image, min_win_size, strid_size):
    """将一个大的图转换成一堆小图像，返回一个位置信息和图像信息"""
    image_height, image_width = image.shape[0], image.shape[1]
    w_height, w_width = min_win_size[0], min_win_size[1]
    s_height, s_width = strid_size[0], strid_size[1]
    h_seq = [i * s_height for i in range(image_height // s_height + 1)]
    w_seq = [i * s_width for i in range(image_width // s_width + 1)]

    candidate_windows, anchorlist = [], []
    for h_st in h_seq:
        for w_st in w_seq:
            h_ed = h_st + w_height
            w_ed = w_st + w_width
            if h_st >= image_height or w_st >= image_width:
                continue
            if min(image_height, h_ed) - h_st < w_height // 2:
                continue
            if min(image_width, w_ed) - w_st < w_width // 2:
                continue
            sub_image = image[h_st:h_ed, w_st:w_ed]
            anchor = [h_st, w_st, h_ed - h_st + 1, w_ed - w_st + 1]
            sub_image.astype(np.float32)
            candidate_windows.append(sub_image)
            anchorlist.append(anchor)
    return candidate_windows, anchorlist

def after_predict(image, anchorlist, predict, color=(255, 0, 0), is_show=False):
    """经过分类模型处理过之后的数据，得到指定区域的概率值，开始绘制热力图"""
    image_height, image_width = image.shape[0], image.shape[1]
    print(type(predict), predict.shape)
    # class_idx = np.argmax(predict, axis=1)
    for idx in range(len(predict)):
        if predict[idx][0] < predict[idx][1]:
            area = anchorlist[idx]
            cv2.rectangle(image, (area[1], area[0]),
                          (area[1]+area[3], area[0]+area[2]), color, 2)
        else:
            area = anchorlist[idx]
            # cv2.rectangle(image, (area[0], area[1]),
            #               (area[0]+area[2], area[1]+area[3]), (0, 255, 0), 1)
    if is_show:
        cv2.imshow("e", image)
        cv2.waitKey()
    return image


if __name__ == "__main__":
    model_server = ModelServer(use_model_flag=1)
    model_server.load_model()

    with open(images_path, 'r') as handle:
        for line in handle.readlines():
            image_path = line.strip()
            image_name = image_path.split("/")[-1]
            if not os.path.exists(image_path):
                raise IOError(image_path + " not found !")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(image.shape)
            # 将遥感图像的大图切成小图，进行识别
            min_win_size, strid_size = (224, 224), (224, 224)
            small_imgs, small_anchors = \
                large2smalls_images(image, min_win_size, strid_size)

            # 对输入图像进行分类器的识别
            small_imgs = [image] # 输入原始图像
            min_win_size, strid_size = (48, 48), (16, 16)
            for sub_img in small_imgs:
                candidate_windows, anchorlist = \
                    prepare_sliding_windows(sub_img, min_win_size, strid_size)
                candidate_windows = np.array(candidate_windows)
                print(candidate_windows.shape)
                st = time.time()
                predict = model_server.predict(candidate_windows)
                ed = time.time()
                print("spend %s s" % str(ed - st))
                after_predict(sub_img, anchorlist, predict, is_show=True)
                # break
            break