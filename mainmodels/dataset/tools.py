# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import json

import os
import random
import xml.dom.minidom
import cv2

from lxml.etree import Element, SubElement, tostring


SampleStep = namedtuple("SampleStep", ["width", "height"])
Point = namedtuple("Point", ["x", "y"])
Rectangle = namedtuple("Rectangle", ["left_up", "right_down"])
BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
RANDOM_SAMPLE_NUM = 500  # 随机采样500张原始图像用来生成训练样本


# 采用蓄水池采样算法对序列进行采样
def rand_selected_file(file_list, K=RANDOM_SAMPLE_NUM):
    res = list()
    for i in range(0, len(file_list)):
        if i < K:
            res.append(file_list[i])
        else:
            M = random.randint(0, i)
            if M < K:
                res[M] = file_list[i]
    return res


def show_image_with_annotation(abs_file_path, target_info_dict):
    image = cv2.imread(abs_file_path)
    for label, target_info_list in target_info_dict.items():
        for target_info in target_info_list:
            print(target_info)
            xmin = int(target_info.xmin)
            ymin = int(target_info.ymin)
            xmax = int(target_info.xmax)
            ymax = int(target_info.ymax)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0))
    cv2.imshow("test", image)
    cv2.waitKey()


def is_region_valid(region):
    if not isinstance(region, Rectangle):
        raise TypeError("|%s| must be Rectangle Type !" % str(region))
    if region.left_up.x < 0 or region.left_up.y < 0:
        return False
    if region.right_down.x < 0 or region.right_down.y < 0:
        return False
    if region.right_down.x < region.left_up.x:
        return False
    if region.right_down.y < region.left_up.y:
        return False
    return True


def calc_region_area(region):
    if not isinstance(region, Rectangle):
        raise TypeError("|%s| must be Rectangle Type !" % str(region))
    print(region)
    return ((region.right_down.x - region.left_up.x) *
            (region.right_down.y - region.left_up.y))


def bbox_2_rectangle(bbox):
    if not isinstance(bbox, BBox):
        raise TypeError("|%s| must be BBox Type !" % str(bbox))
    return Rectangle(Point(bbox.xmin, bbox.ymin), Point(bbox.xmax, bbox.ymax))


def is_region_inner(larger_region, small_region):
    flag_left = (larger_region.left_up.x <= small_region.left_up.x and
                 larger_region.left_up.y <= small_region.left_up.y)
    flag_right = (larger_region.right_down.x >= small_region.right_down.x
                  and larger_region.right_down.y >= small_region.right_down.y)
    if flag_left and flag_right:
        return True
    return False


def is_region_overlap(region01, region02):
    if not (
        isinstance(region01, Rectangle) and isinstance(region02, Rectangle)):
        raise TypeError("region0x must be Rectangle Type !")
    left_up_pos = Point(
        max(region01.left_up.x, region02.left_up.x),
        max(region01.left_up.y, region02.left_up.y))
    right_down_pos = Point(
        min(region01.right_down.x, region02.right_down.x),
        min(region01.right_down.y, region02.right_down.y))
    return (left_up_pos.x < right_down_pos.x and
            left_up_pos.y < right_down_pos.y)


def region_overlap_area(rect01, rect02):
    """计算两个区域的重叠的面积"""
    if not (isinstance(rect01, Rectangle) and isinstance(rect02, Rectangle)):
        raise TypeError("rect0x must be Rectangle Type !")

    if not is_region_overlap(rect01, rect02):
        return -1

    left_up_pos = Point(
        max(rect01.left_up.x, rect02.left_up.x),
        max(rect01.left_up.y, rect02.left_up.y))
    right_down_pos = Point(
        min(rect01.right_down.x, rect02.right_down.x),
        min(rect01.right_down.y, rect02.right_down.y))
    return calc_region_area(Rectangle(left_up_pos, right_down_pos))


class UtilityTools(object):
    @staticmethod
    def ergodic_crop_region(src_region, target_height, target_width, step=None):
        if not isinstance(src_region, Rectangle):
            raise IOError("src_region must be Rectangle Type !")
        left_up = src_region.left_up
        right_down = src_region.right_down
        src_height = src_region.right_down.y - src_region.left_up.y
        src_width = src_region.right_down.x - src_region.left_up.x
        if src_height < target_height or src_width < target_width:
            raise IOError("output region larger the source region !")
        if step is None:
            step = SampleStep(1, 1)
        if not isinstance(step, SampleStep):
            raise IOError("step must be SampleStep Type !")
        rect_list = []
        step_width, step_height = step
        for st_y in range(left_up.y, right_down.y - target_height + 1,
                          step_height):
            for st_x in range(left_up.x, right_down.x - target_width + 1,
                              step_width):
                t_left_pos = Point(st_x, st_y)
                t_right_pos = Point(st_x + target_width, st_y + target_height)
                t_rect = Rectangle(t_left_pos, t_right_pos)
                rect_list.append(t_rect)
        return rect_list

    @staticmethod
    def ergodic_crop_region_with_transform(
            src_region, object_pos_dict, target_height, target_width,
            step=None):
        """
        :param src_region: 用户输入的原始图像的大小
        :param object_pos_dict: 全局目标object_pos_dict = {"label": [BBox, BBox]}
        :param step: SampleStep(width, height)
        """
        if not isinstance(src_region, Rectangle):
            raise IOError("src_region must be Rectangle Type !")
        left_up = src_region.left_up
        right_down = src_region.right_down
        src_height = src_region.right_down.y - src_region.left_up.y
        src_width = src_region.right_down.x - src_region.left_up.x
        if src_height < target_height or src_width < target_width:
            raise IOError("output region larger the source region !")
        if step is None:
            step = SampleStep(1, 1)
        if not isinstance(step, SampleStep):
            raise IOError("step must be SampleStep Type !")

        rect_objects_dict = dict()
        step_width, step_height = step
        for st_y in range(left_up.y, right_down.y - target_height + 1,
                          step_height):
            for st_x in range(left_up.x, right_down.x - target_width + 1,
                              step_width):
                t_left_pos = Point(st_x, st_y)
                t_right_pos = Point(st_x + target_width, st_y + target_height)
                t_rect = Rectangle(t_left_pos, t_right_pos)
                transformed_pos_dict = UtilityTools.transform_global_object_pos(
                    t_rect, object_pos_dict)
                if transformed_pos_dict:
                    rect_objects_dict[t_rect] = transformed_pos_dict
        return rect_objects_dict

    @staticmethod
    def transform_global_object_pos(small_region, object_pos_dict):
        """可以将用户输入的目标的位置信息转换到指定的区域的局部坐标
        large_region (0, 0, 100, 100) -> small_region (40, 40, 80, 80)
        object_pos (50, 50, 60 ,60) -> (10, 10, 20, 20)
        case 01:
            物体的位置转换之后是非法的
        case 02:
            物体的位置转换之后完全不在给定的small_region区域
        case 03:
            物体的位置转换之后超过80%给定的区域
        """
        delta_width = small_region.left_up.x
        delta_height = small_region.left_up.y
        l_x, l_y = 0, 0
        r_x = small_region.right_down.x - delta_width
        r_y = small_region.right_down.y - delta_height
        transformed_small_region = Rectangle(Point(l_x, l_y), Point(r_x, r_y))
        transformed_pos_dict = {}
        for label, object_pos_list in object_pos_dict.items():
            transformed_list = []
            for bbox in object_pos_list:
                new_pos = BBox(bbox.xmin - small_region.left_up.x,
                               bbox.ymin - small_region.left_up.y,
                               bbox.xmax - small_region.left_up.x,
                               bbox.ymax - small_region.left_up.y)
                region_bbox = bbox_2_rectangle(new_pos)
                if not is_region_valid(region_bbox):
                    continue
                # TODO(liuguiyang): 对于候选区域是否在给定的小区域中
                if is_region_inner(transformed_small_region, region_bbox):
                    transformed_list.append(new_pos)
            if transformed_list:
                transformed_pos_dict[label] = transformed_list
        return transformed_pos_dict


# 给定一个标记文件，找到对应的目标的位置信息
def extract_airplane_posinfo(filename):
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

def write_xml_format(abs_img_path, abs_anno_path, anno_list, img_format="jpg"):
    src_img_data = cv2.imread(abs_img_path)
    img_height, img_width, img_channle = src_img_data.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'CSUVideo'
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

    for anno_target in anno_list:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
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
    tmp = abs_anno_path.split("/")[-1]
    tmp = ".".join(tmp.split(".")[:-1] + ["xml"])
    with open(abs_anno_path, "w") as handler:
        handler.write(xml_obj)
    # tmp = abs_img_path.split("/")[:-1]
    # cv2.imwrite(save_dir + "JPEGImages/" + tmp, src_img_data)

def fetch_xml_format(src_img_data, f_name, anno_list):
    img_height, img_width, img_channle = src_img_data.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'CSUVideo'
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

if __name__ == '__main__':
    left_pos = Point(0, 0)
    right_pos = Point(5, 4)
    x, y = right_pos
    rect = Rectangle(left_pos, right_pos)

    src_region = Rectangle(Point(0, 0), Point(2048, 2048))
    target_height = 480
    target_width = 640
    step = SampleStep(width=100, height=100)
    tmp = UtilityTools.ergodic_crop_region(
        src_region, target_height, target_width, step)
    small_region = Rectangle(Point(3, 4), Point(10, 20))
    object_pos_dict = {
        "l1": [BBox(5, 5, 8, 8)]
    }
    UtilityTools.transform_global_object_pos(small_region, object_pos_dict)
    src_region = Rectangle(Point(0, 0), Point(4, 5))
    print(region_overlap_area(src_region, small_region))
