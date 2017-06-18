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
from PIL import Image

from mainmodels.dataset.tools import show_image_with_annotation
from mainmodels.dataset.tools import UtilityTools
from mainmodels.dataset.tools import Point
from mainmodels.dataset.tools import SampleStep
from mainmodels.dataset.tools import Rectangle
from mainmodels.dataset.tools import BBox

TRAIN_ANNOTATION_SAVE_PATH = "/Volumes/projects/TrafficSign/Tencent-Tsinghua" \
                       "/StandardData/train_annotation.json"
TEST_ANNOTATION_SAVE_PATH = "/Volumes/projects/TrafficSign/Tencent-Tsinghua" \
                       "/StandardData/test_annotation.json"
TRAIN_DATASET_SAVE_DIR = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/StandardData/train"
TEST_DATASET_SAVE_DIR = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/StandardData/test"


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

    def __init_annotation(self, recreated=False):
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
        target_width, target_height = 400, 260
        if recreated:
            print("开始采样训练数据")
            dataset_dict = self.__sample_samples(
                self._train_set, target_height, target_width)
            self.__save_crop_images(dataset_dict, TRAIN_DATASET_SAVE_DIR,
                                    TRAIN_ANNOTATION_SAVE_PATH)

            print("开始采样测试数据")
            dataset_dict = self.__sample_samples(
                self._test_set, target_height, target_width)
            self.__save_crop_images(dataset_dict, TEST_DATASET_SAVE_DIR,
                                    TEST_ANNOTATION_SAVE_PATH)


    def __sample_samples(self, dataset, target_height, target_width):
        """dataset_dict
        {
        '/Volumes/projects/TrafficSign/Tencent-Tsinghua/data/train/10926.jpg': {
            Rectangle(left_up=Point(x=750, y=760), right_down=Point(x=1150, y=1020)):
                {'pn': [BBox(xmin=376.3699999999999, ymin=231.54200000000003, xmax=387.80849999999987, ymax=242.9851)]},
            Rectangle(left_up=Point(x=750, y=780), right_down=Point(x=1150, y=1040)):
                {'pn': [BBox(xmin=376.3699999999999, ymin=211.54200000000003, xmax=387.80849999999987, ymax=222.9851)]},
            Rectangle(left_up=Point(x=750, y=800), right_down=Point(x=1150, y=1060)):
                {'pn': [BBox(xmin=376.3699999999999, ymin=191.54200000000003, xmax=387.80849999999987, ymax=202.9851)]}
        }
        """
        dataset_dict = dict()
        for file_path, target_list in dataset.items():
            abs_file_path = "/".join([self._base_dir, file_path])
            image = Image.open(abs_file_path)
            img_width, img_height = image.size
            # print(img_width, img_height)
            # pprint.pprint(target_list)
            object_pos_dict = {}
            for cell_object_dict in target_list:
                bbox_dict = cell_object_dict["bbox"]
                bbox = BBox(bbox_dict["xmin"], bbox_dict["ymin"],
                            bbox_dict["xmax"], bbox_dict["ymax"])
                object_pos_dict.setdefault(cell_object_dict["category"], [])
                object_pos_dict[cell_object_dict["category"]].append(bbox)

            # pprint.pprint(object_pos_dict)
            left_up = Point(0, 0)
            right_down = Point(img_width, img_height)
            src_region = Rectangle(left_up, right_down)
            sample_step = SampleStep(width=50, height=20)
            rect_objects_dict = UtilityTools.ergodic_crop_region_with_transform(
                src_region, object_pos_dict, target_height, target_width,
                sample_step)
            # pprint.pprint(rect_objects_dict)
            dataset_dict[abs_file_path] = rect_objects_dict
            # pprint.pprint(dataset_dict)
        return dataset_dict

    def __save_crop_images(self, dataset_dict, save_dir, annotation_file_path):
        """dataset_dict
            {
            '/Volumes/projects/TrafficSign/Tencent-Tsinghua/data/train/10926.jpg': {
                Rectangle(left_up=Point(x=750, y=760), right_down=Point(x=1150, y=1020)):
                    {'pn': [BBox(xmin=376.3699999999999, ymin=231.54200000000003, xmax=387.80849999999987, ymax=242.9851)]},
                Rectangle(left_up=Point(x=750, y=780), right_down=Point(x=1150, y=1040)):
                    {'pn': [BBox(xmin=376.3699999999999, ymin=211.54200000000003, xmax=387.80849999999987, ymax=222.9851)]},
                Rectangle(left_up=Point(x=750, y=800), right_down=Point(x=1150, y=1060)):
                    {'pn': [BBox(xmin=376.3699999999999, ymin=191.54200000000003, xmax=387.80849999999987, ymax=202.9851)]}
            }
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        annotation_dict = dict()
        for abs_file, object_pos_dict in dataset_dict.items():
            if not os.path.exists(abs_file):
                continue
            src_image = cv2.imread(abs_file)
            # print(object_pos_dict)
            for t_rect, cell_dict in object_pos_dict.items():
                l_x, l_y = t_rect.left_up.x, t_rect.left_up.y
                r_x, r_y = t_rect.right_down.x, t_rect.right_down.y
                crop_image = src_image[l_y:r_y, l_x:r_x]
                image_name = os.path.basename(abs_file)
                image_name = ".".join(image_name.split(".")[0:-1])
                image_name += "_%d_%d_%d_%d" % (int(l_x), int(l_y),
                                                int(r_x), int(r_y))
                image_name += ".png"
                save_abs_file_path = "/".join([save_dir, image_name])
                cv2.imwrite(save_abs_file_path, crop_image)
                # show_image_with_annotation(save_abs_file_path, cell_dict)
                annotation_dict[save_abs_file_path] = cell_dict

        with open(annotation_file_path, "w") as handle:
            handle.write(json.dumps(
                annotation_dict, indent=4, sort_keys=False, ensure_ascii=False))


if __name__ == '__main__':
    obj_dataset = TT100K_DataSet()
    class_types = obj_dataset.class_types
    class_dict = dict()
    for idx, label in enumerate(class_types):
        class_dict[label] = idx+1
    with open("TT100K_traffic_sign.json", "w") as handle:
        handle.write(json.dumps(
            class_dict, indent=4, sort_keys=False, ensure_ascii=False))
    # obj_dataset.load_annotation()