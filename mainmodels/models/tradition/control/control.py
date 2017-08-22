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
import copy
import codecs

import cv2
import numpy as np

# import __init

from mainmodels.dataset.airplane_input import extract_airplane_posinfo
from mainmodels.models.tradition.control.model_cnn_server import ModelServer


proj_root = "/".join(os.path.abspath(__file__).split("/")[0:-5])
images_path = "/".join(["/Volumes/projects/第三方数据下载/JL1ST", "images.list"])
res_save_dir = "/Volumes/projects/第三方数据下载/train_log/res_images"

# [h, w]
# target_size_dict = {
#     "airplane": [[20, 22], [25, 25], [32, 32], [50, 50], [65, 65], [65, 70],
#                  [65, 75], [65, 80], [70, 70], [70, 75], [75, 75], [75, 80],
#                  [80, 80], [80, 85], [90, 75], [90, 85], [110, 100],
#                  [110, 110]],
#     "ship": [[40, 30], [70, 60], [70, 70], [70, 90], [70, 115], [100, 115]],
#     "storage tank": [[15, 15], [20, 20], [30, 30],
#                      [40, 40], [50, 50], [65, 65]],
#     "harbor": [[75, 90], [100, 110], [125, 140], [150, 145]],
#     "bridge": [[25, 25], [30, 25], [40, 25], [50, 25], [60, 30], [70, 40],
#                [45, 65], [50, 90]]
# }

target_size_dict = {
    "airplane": [[78, 65], [98, 85], [70, 95]],
    "ship": [[40, 30], [70, 60], [70, 70], [70, 115], [100, 115]],
    "storage tank": [[15, 15], [20, 20], [30, 30],
                     [40, 40], [50, 50], [65, 65]],
    "harbor": [[75, 90], [100, 110], [125, 140], [150, 145]],
    "bridge": [[30, 25], [40, 25], [50, 25], [60, 30], [70, 40],
               [45, 65], [50, 90]],
    "total": [[48, 48], [56, 56]]
}

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

def prepare_sliding_windows_new(image, min_win_size):
    """准备滑动窗口数据"""
    image_height, image_width = image.shape[0], image.shape[1]
    # if image.shape[2] == 3:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    w_height, w_width = min_win_size[0], min_win_size[1]
    s_height, s_width = w_height // 3 * 1, w_width // 3 * 1
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
            IMG_H, IMG_W = 56, 56
            if sub_image.shape[0] != IMG_H or sub_image.shape[1] != IMG_W:
                sub_image = cv2.resize(sub_image, (IMG_H, IMG_W),
                                       interpolation=cv2.INTER_LINEAR)
            sub_image = sub_image.reshape((IMG_H, IMG_W, 1))
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

def calc_softmax_prob(predict):
    import math
    res = []
    for item in predict:
        tmp = []
        for val in item:
            tmp.append(math.exp(val))
        tmp_sum = sum(tmp)
        for i in range(len(tmp)):
            tmp[i] = tmp[i] / tmp_sum
        res.append(tmp)
    return np.array(res)


def show_ground_truth(image_name, src_image):
    anno_name = ".".join(image_name.split(".")[:-1] + ["xml"])
    annotation_file = \
        "/Volumes/projects/第三方数据下载/JL1ST/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_annotation/"
    abs_anno_file = annotation_file + anno_name
    print(abs_anno_file)
    if not os.path.exists(abs_anno_file):
        return src_image
    ground_truths = extract_airplane_posinfo(abs_anno_file)
    color = (0, 0, 255)
    for area in ground_truths:
        # cv2.rectangle(src_image, (area[1], area[0]),(area[3], area[2]), color, 2)
        print(area)
        cv2.rectangle(src_image,
                      (area[2], area[3]), (area[0], area[1]), color, 2)
        src_image = cv2.putText(src_image, "airplane",
                            (area[2], area[3]), 0,
                            0.5, color, 1, cv2.LINE_AA)
    return src_image


def after_predict(image_name, min_win_size, src_image, anchorlist, predict,
                  color=(255, 0, 0), is_show=False):
    """经过分类模型处理过之后的数据，得到指定区域的概率值，开始绘制热力图"""
    image = copy.deepcopy(src_image)
    image_height, image_width = image.shape[:2]
    print(type(predict), predict.shape)
    predict = calc_softmax_prob(predict)
    print(predict)
    class_idx = np.argmax(predict, axis=1)
    for idx in range(len(predict)):
        if class_idx[idx] == 0:
            continue
        if min_win_size == [45, 45]:
            if class_idx[idx] in [1, 3] and predict[idx][class_idx[idx]] >= 0.92:
                area = anchorlist[idx]
                cv2.rectangle(image, (area[1], area[0]),
                              (area[1] + area[3], area[0] + area[2]), color, 2)
                image = cv2.putText(image, str(class_idx[idx]),
                                    (area[1], area[0]), 0,
                                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            b1_flag = class_idx[idx] in [1, 2, 3] and predict[idx][class_idx[
                idx]] >= 0.00
            b2_flag = class_idx[idx] in [4, 5] and predict[idx][class_idx[idx]] >= 0.9

            if b1_flag or b2_flag:
                area = anchorlist[idx]
                cv2.rectangle(image, (area[1], area[0]),
                              (area[1]+area[3], area[0]+area[2]), color, 2)
                image = cv2.putText(image, str(class_idx[idx]),
                                (area[1], area[0]), 0,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    image = show_ground_truth(image_name, image)
    if is_show:
        cv2.imshow("e", image)
        cv2.waitKey()
    else:
        save_file_path = "/".join(
            [res_save_dir, image_name[:-4] + "_" +
             str(min_win_size[0])+"_"+str(min_win_size[1]) + ".png"])
        cv2.imwrite(save_file_path, image)
    return image


if __name__ == "__main__":
    model_server = ModelServer(use_model_flag=0)
    model_server.load_model()

    with codecs.open(images_path, 'r', "utf8") as handle:
        for line in handle.readlines():
            image_path = line.strip()
            image_name = image_path.split("/")[-1]
            if not os.path.exists(image_path):
                raise IOError(image_path + " not found !")
            image = cv2.imread(image_path, 0)
            print(image.shape)
            # 将遥感图像的大图切成小图，进行识别
            # min_win_size, strid_size = (224, 224), (224, 224)
            # small_imgs, small_anchors = \
            #     large2smalls_images(image, min_win_size, strid_size)

            # 对输入图像进行分类器的识别, 输入原始图像
            small_imgs = [image]
            for sub_img in small_imgs:
                preds = []
                anchor_lists = []
                min_win_size = (56, 56)
                for min_win_size in target_size_dict["total"]:
                    candidate_windows, anchorlist = \
                        prepare_sliding_windows_new(sub_img, min_win_size)
                    candidate_windows = np.array(candidate_windows)
                    print(candidate_windows.shape)
                    st = time.time()
                    predict = model_server.predict(candidate_windows)
                    ed = time.time()
                    print("spend %s s" % str(ed - st))
                    preds.append(predict)
                    anchor_lists.extend(anchorlist)

                # get rgb src image and label the target into the image
                predict = np.concatenate((preds[0], preds[1]), axis=0)
                rgb_image = cv2.imread(image_path)
                after_predict(image_name, min_win_size,
                              rgb_image, anchor_lists, predict, is_show=False)
                # break
            # break