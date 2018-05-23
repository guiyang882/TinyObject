# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/9/25

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 将其他数据格式的数据转换成VOC标准格式

import os
import json
from lxml.etree import Element, SubElement, tostring
import cv2


def fetch_label(label_map_path):
    with open(label_map_path, "r") as handler:
        label2id_dict = json.load(handler)
    id2label_dict = dict()
    for key, val in label2id_dict.items():
        id2label_dict[val] = key
    return id2label_dict

id2label_dict = fetch_label("/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/sub_traffic_sign.json")


def write_xml_format(abs_img_path, abs_anno_path, save_dir, img_format="jpg"):
    src_img_data = cv2.imread(abs_img_path)
    img_height, img_width, img_channle = src_img_data.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'TT100K'
    node_filename = SubElement(node_root, 'filename')
    if abs_img_path.split(".")[-1] != img_format:
        abs_img_path = ".".join(abs_img_path.split(".")[:-1] + [img_format])
    node_filename.text = abs_img_path.split("/")[-1]

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(img_channle)

    with open(abs_anno_path, "r") as handler:
        anno_list = json.load(handler)
        for anno_target in anno_list:
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = id2label_dict[anno_target[-1]]
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(anno_target[0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(anno_target[1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(anno_target[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(anno_target[3])
    xml_obj = tostring(node_root, pretty_print=True)
    xml_obj = xml_obj.decode("utf8")
    tmp = abs_anno_path.split("/")[-1]
    tmp = ".".join(tmp.split(".")[:-1] + ["xml"])
    print(save_dir + "Annotations/" + tmp)
    with open(save_dir + "Annotations/" + tmp, "w") as handler:
        handler.write(xml_obj)
    tmp = abs_img_path.split("/")[:-1]
    cv2.imwrite(save_dir + "JPEGImages/" + tmp, src_img_data)


def convert2xml(img_postfix="png", anno_postfix="json"):
    img_dir = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/VOCFORMAT/JPEGImages/"
    annotation_label_dir = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/VOCFORMAT/Annotations/"
    save_dir = "/home/ai-i-liuguiyang/repos_ssd/Tencent-Tsinghua/standard_data/TT100K/"
    annotation_lists = os.listdir(annotation_label_dir)
    for anno_name in annotation_lists:
        if anno_name.startswith("._") or anno_name.startswith("."):
            continue
        if not anno_name.endswith(anno_postfix):
            continue
        img_name = ".".join(anno_name.split(".")[:-1] + [img_postfix])
        img_path = img_dir + img_name
        if not os.path.exists(img_path):
            continue
        write_xml_format(img_path, annotation_label_dir + anno_name, save_dir)
        break


if __name__ == '__main__':
    convert2xml()