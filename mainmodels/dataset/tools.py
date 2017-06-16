# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple


SampleStep = namedtuple("SampleStep", ["width", "height"])
Point = namedtuple("Point", ["x", "y"])
Rectangle = namedtuple("Rectangle", ["left_up", "right_down"])
BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])


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
    if not (isinstance(region01, Rectangle) and isinstance(region02, Rectangle)):
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
        for st_y in range(left_up.y, right_down.y-target_height+1, step_height):
            for st_x in range(left_up.x, right_down.x-target_width+1, step_width):
                t_left_pos = Point(st_x, st_y)
                t_right_pos = Point(st_x+target_width, st_y+target_height)
                t_rect = Rectangle(t_left_pos, t_right_pos)
                rect_list.append(t_rect)
        return rect_list

    @staticmethod
    def ergodic_crop_region_with_transform(
        src_region, object_pos_dict, target_height, target_width, step=None):
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
        for st_y in range(left_up.y, right_down.y-target_height+1, step_height):
            for st_x in range(left_up.x, right_down.x-target_width+1, step_width):
                t_left_pos = Point(st_x, st_y)
                t_right_pos = Point(st_x+target_width, st_y+target_height)
                t_rect = Rectangle(t_left_pos, t_right_pos)
                rect_list.append(t_rect)
        return rect_list

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
        for label, object_pos_list in object_pos_dict.items():
            transformed_list = []
            for bbox in object_pos_list:
                new_pos = BBox(bbox.xmin-small_region.left_up.x,
                               bbox.ymin-small_region.left_up.y,
                               bbox.xmax-small_region.left_up.x,
                               bbox.ymax-small_region.left_up.y)
                region_bbox = bbox_2_rectangle(new_pos)
                if not is_region_valid(region_bbox):
                    continue
                # TODO(liuguiyang): 对于候选区域是否在给定的小区域中
                transformed_list.append(new_pos)
            object_pos_dict[label] = transformed_list
        print(object_pos_dict)
        return object_pos_dict


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
