# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import cv2

import xml.dom.minidom
from lxml.etree import Element, SubElement, tostring
from mainmodels.dataset.VEDAI import vedia_config as config

# 提取图像对应的标注的数据
def fetch_anno_targets_info(abs_anno_path, is_label_text=False):
    if not os.path.exists(abs_anno_path):
        raise IOError("No Such annotation file !")
    with open(abs_anno_path, "r") as anno_reader:
        total_annos = list()
        for line in anno_reader:
            line = line.strip()
            sub_anno = re.split("\(|\,|\)", line)
            a = [int(item) for item in sub_anno if len(item)]
            if len(a) == 5:
                if is_label_text:
                    total_annos.append(a[:4]+[config.idx_sign_dict[a[-1]]])
                else:
                    total_annos.append(a)
        return total_annos

def fetch_xml_format(src_img_data, f_name, anno_list):
    img_height, img_width, img_channle = src_img_data.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VEDAI'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(img_channle)

    for anno_target in anno_list:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        if isinstance(anno_target[-1], int) and anno_target[-1] in config.remove_label_ids:
            continue
        if isinstance(anno_target[-1], int):
            node_name.text = config.idx_sign_dict[anno_target[-1]]
        else:
            node_name.text = anno_target[-1]
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
    return xml_obj

# 给定一个标记文件，找到对应的目标的位置信息
def extract_target_from_xml(filename):
    if not os.path.exists(filename):
        raise IOError(filename + " not exists !")
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    # 获取集合中所有的目标
    targets = collection.getElementsByTagName("object")
    res = []
    for target in targets:
        target_name = target.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = target.getElementsByTagName("bndbox")[0]
        xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data
        res.append([int(xmin), int(ymin), int(xmax), int(ymax), target_name])
    return res

# 原始数据中多目标的显示
def show_targets(img_dir, anno_dir):
    for img_name in os.listdir(img_dir):
        if img_name.startswith("._"):
            continue
        abs_img_path = img_dir+img_name
        abs_anno_path = anno_dir+img_name.replace("jpg", "xml")
        target_annos = extract_target_from_xml(abs_anno_path)
        image = cv2.imread(abs_img_path)
        for target_info in target_annos:
            xmin, ymin, xmax, ymax = target_info[:4]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.imshow("src", image)
        cv2.waitKey()

def fetch_targets_info(abs_anno_path):
    anno_targets = list()
    normal_anno_targets = list()
    with open(abs_anno_path, "r") as anno_reader:
        for line in anno_reader:
            line = line.strip().split(" ")
            if len(line) != 14:
                continue
            label_id = int(line[3])
            # x1,x2,x3,x4,y1,y2,y3,y4
            t = [int(a) for a in line[-8:]]
            coordinate_pos = [t[0], t[4], t[1], t[5], t[2], t[6], t[3], t[7]]
            normal_pos = [min(t[:4]), min(t[4:8]), max(t[:4]), max(t[4:8])]
            anno_targets.append(coordinate_pos + [label_id])
            normal_anno_targets.append(normal_pos + [label_id])
    return anno_targets, normal_anno_targets


if __name__ == '__main__':
    a = fetch_anno_targets_info(
        "/Volumes/projects/repos/RSI/NWPUVHR10/sub_annotation/001.txt")
    print(a)