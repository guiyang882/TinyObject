# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import pprint

import cv2


class TT100K_DataSet(object):

    def __init__(self):
        self._base_dir = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/data"
        self._annotations_path = "/".join([self._base_dir, "annotations.json"])
        self._train_set = dict()
        self._test_set = dict()
        self._other_set = dict()
        self.__init_annotation()

    @staticmethod
    def __is_file_exits(file_path):
        if not os.path.exists(file_path):
            raise IOError("%s not exitst !" % file_path)

    @property
    def class_nums(self):
        return self.__class_nums

    @property
    def image_nums(self):
        return self.__image_nums

    @property
    def class_types(self):
        return self._annotations_data["types"]

    @property
    def nums_other_class(self):
        return len(self._other_set)

    @property
    def nums_test_class(self):
        return len(self._test_set)

    @property
    def nums_train_class(self):
        return len(self._train_set)

    def __init_annotation(self):
        self.__is_file_exits(self._annotations_path)
        with open(self._annotations_path, 'r') as handle:
            self._annotations_data = json.load(handle)
        self.__class_nums = len(self._annotations_data["types"])
        self.__image_nums = len(self._annotations_data["imgs"])
        for file_id, file_dict in self._annotations_data["imgs"].items():
            if "other" in file_dict["path"]:
                self._other_set[file_dict["path"]] = file_dict["objects"]
            elif "test" in file_dict["path"]:
                self._test_set[file_dict["path"]] = file_dict["objects"]
            elif "train" in file_dict["path"]:
                self._train_set[file_dict["path"]] = file_dict["objects"]
            else:
                print(file_dict["path"])

        for file_path, target_list in self._train_set.items():
            abs_file_path = "/".join([self._base_dir, file_path])
            pprint.pprint(target_list[0])
            self.__show_image_with_annotation(abs_file_path, target_list)
            break

    @staticmethod
    def __show_image_with_annotation(abs_file_path, target_info_list):
        image = cv2.imread(abs_file_path)
        for target_info in target_info_list:
            xmin = int(target_info["bbox"]["xmin"])
            ymin = int(target_info["bbox"]["ymin"])
            xmax = int(target_info["bbox"]["xmax"])
            ymax = int(target_info["bbox"]["ymax"])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0))
        cv2.imshow("test", image)
        cv2.waitKey()

    def __prepare_train_data(self):
        pass

if __name__ == '__main__':
    obj_dataset = TT100K_DataSet()
    # obj_dataset.load_annotation()