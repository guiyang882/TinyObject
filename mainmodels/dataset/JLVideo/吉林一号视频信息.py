# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
主要是统计吉林一号四段视频数据中的飞机信息
'''

import os

import cv2

from mainmodels.dataset import tools

base_dir = "/Volumes/projects/repos/RSI/"
video_name = "large_000013363_total"
video_name = "large_000014631_total"
video_name = "large_minneapolis_1_total"
video_name = "large_tunisia_total"
sub_annotation_dir = base_dir + video_name + "/sub_annotation/"
annotation_dir = base_dir + video_name + "/annotation/"
if not os.path.exists(annotation_dir):
    os.makedirs(annotation_dir)
frame_dir = base_dir + video_name + "/large_src/"
images_prefix = list()


# 统计出一张大图像中每个目标的数量
# zero_padding: 文件名字的后缀是按照几个零补齐占位的
def count_large_target_info(zero_padding=4):
    max_idx = 0
    for item in os.listdir(sub_annotation_dir):
        max_idx = max(max_idx, int(item.strip().split(".")[0].split("_")[-1]))
        tmp_prefix = "_".join(item.strip().split(".")[0].split("_")[:4]) + "_"
        if tmp_prefix not in images_prefix:
            images_prefix.append(tmp_prefix)
    print(images_prefix)

    targets_list = list()
    for idx in range(1, max_idx+1):
        total_target_info = dict()
        for prefix in images_prefix:
            anno_path_tpl = sub_annotation_dir + prefix + "%0"+str(zero_padding)+"d.xml"
            anno_path = anno_path_tpl % idx
            if not os.path.exists(anno_path):
                continue
            sub_target_info = tools.extract_airplane_posinfo(anno_path)
            total_target_info[prefix] = sub_target_info
        tmp = 0
        for key, val in total_target_info.items():
            # print(key, len(val))
            tmp += len(val)
        if tmp:
            targets_list.append(tmp)
    print(targets_list)

# 将部分图像的标注数据显示在原始的图像中
def show_annotation_src(anno_zero_padding=4, frame_zero_padding=6):

    def _prefix2position(name_prefix):
        '''
        width, height = frame.shape[0], frame.shape[1]
        c_x, c_y = width // 2, height // 2
        [0:c_x, 0:c_y]
        '''
        position = [int(a) for a in name_prefix.strip('_').split('_')]
        return position[0], position[2]

    max_idx = 0
    for item in os.listdir(sub_annotation_dir):
        max_idx = max(max_idx, int(item.strip().split(".")[0].split("_")[-1]))
        tmp_prefix = "_".join(item.strip().split(".")[0].split("_")[:4]) + "_"
        if tmp_prefix not in images_prefix:
            images_prefix.append(tmp_prefix)

    for idx in range(1, max_idx + 1):
        total_target_info = dict()
        for prefix in images_prefix:
            anno_path_tpl = sub_annotation_dir + prefix + "%0" + str(
                anno_zero_padding) + "d.xml"
            anno_path = anno_path_tpl % idx
            if not os.path.exists(anno_path):
                continue
            # [xmin, ymin, xmax, ymax]
            sub_target_info = tools.extract_airplane_posinfo(anno_path)
            total_target_info[prefix] = sub_target_info
        frame_path_tpl = frame_dir + "%0" + str(frame_zero_padding) + "d.jpg"
        frame_path = frame_path_tpl % idx
        if not os.path.exists(frame_path):
            continue
        src_image = cv2.imread(frame_path)
        anno_list = list()
        for key, val in total_target_info.items():
            height_shift, width_shift = _prefix2position(key)
            for pos in val:
                new_pos = [pos[0]+width_shift, pos[1]+height_shift,
                           pos[2]+width_shift, pos[3]+height_shift]
                anno_list.append(new_pos + ["plane"])
                cv2.rectangle(src_image,
                              (new_pos[0], new_pos[1]), (new_pos[2], new_pos[3]),
                              (0, 255, 0), 2)
        # cv2.imwrite(base_dir+"tmp/%06d.jpg" % idx, src_image)
        abs_anno_path = annotation_dir + "%06d.xml" % idx
        print(abs_anno_path)
        tools.write_xml_format(frame_path, abs_anno_path, anno_list)
        # break

import shutil

# 将转换好的标准数据转换成VOC格式的数据集
def copy2voc(save_prefix):
    if not os.path.exists(save_prefix + video_name):
        os.makedirs(save_prefix + video_name)
    if not os.path.exists(save_prefix + video_name + "/JPEGImages"):
        os.makedirs(save_prefix + video_name + "/JPEGImages")
    if not os.path.exists(save_prefix + video_name + "/Annotations"):
        os.makedirs(save_prefix + video_name + "/Annotations")
    # 根据当前的video_name来拷贝对应的图像和xml文件
    for anno_name in os.listdir(annotation_dir):
        if anno_name.startswith("._"):
            continue
        shutil.copyfile(frame_dir+anno_name.replace("xml", "jpg"),
                        save_prefix+video_name+"/JPEGImages/"+anno_name
                        .replace("xml", "jpg"))
        shutil.copyfile(annotation_dir+anno_name,
                        save_prefix+video_name+"/Annotations/"+anno_name)
        print(anno_name)

if __name__ == '__main__':
    # show_annotation_src()
    copy2voc("/Volumes/projects/repos/RSI/VOCFormat/")
