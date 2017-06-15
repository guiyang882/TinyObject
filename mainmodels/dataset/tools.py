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
        step_width, step_height = step
        for st_y in range(left_up.y, right_down.y+1, step_height):
            for st_x in range(left_up.x, right_down.x+1, step_width):
                pass

    @staticmethod
    def is_region_overlap(region01, region02):
        if (not isinstance(region01, Rectangle) or
                not isinstance(region02, Rectangle)):
            raise IOError("region0x must be Rectangle Type !")
        left_up_pos = Point(
            max(region01.left_up.x, region02.left_up.x),
            max(region01.left_up.y, region02.left_up.y))
        right_down_pos = Point(
            min(region01.right_down.x, region02.right_down.x),
            min(region01.right_down.y, region02.right_down.y))
        return (left_up_pos.x < right_down_pos.x and
                left_up_pos.y < right_down_pos.y)

    @staticmethod
    def is_region_inner(larger_region, small_region):
        pass


if __name__ == '__main__':
    left_pos = Point(0, 0)
    right_pos = Point(5, 4)
    x, y = right_pos
    rect = Rectangle(left_pos, right_pos)
    print(x, y)