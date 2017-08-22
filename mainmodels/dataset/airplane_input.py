# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json, codecs
import random
import xml.dom.minidom
import cv2


# 解析XML文本得到JL1ST的目标数据
dir_prefix = "/Volumes/projects/第三方数据下载/JL1ST/"
JL1ST_NAME = "JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS"
annotation_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_annotation/"
image_dir = dir_prefix + "SRC_" + JL1ST_NAME + "/"
target_save_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_TARGET/"
neg_samples_save_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_NEG_SAMPLES/"
ssd_sample_dir = dir_prefix + "SRC_" + JL1ST_NAME + "_SSD/"
SSD_IMG_W, SSD_IMG_H = 512, 512


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


# 给定一张图像和图像中的位置，存储图像中对应的位置
def saveTarget(imagepath, labelinfo, savedir):
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    if not os.path.exists(imagepath):
        raise IOError(imagepath + " not found !")
    imagename = imagepath.split("/")[-1].split(".")[0]
    img = cv2.imread(imagepath)
    cnt = 0
    for pos in labelinfo:
        cnt += 1
        print(pos)
        xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        # targetname = pos[4]
        # width = ymax - ymin
        # height = xmax - xmin
        subimg = img[ymin:ymax, xmin:xmax]
        if subimg.shape[2] == 3:
            subimg = cv2.cvtColor(subimg, cv2.COLOR_RGB2GRAY)
        savename = imagename + "_" + str(cnt) + ".png"
        cv2.imwrite(savedir+savename, subimg)


# 根据给定的图像和目标的位置，进行负样本的采样
def negative_sample(imagepath, labelinfo, savedir):
    g_overlap_ratio = 0.5  # 正负样本的的重合面积与正样本的比例

    def check_valid(neg_rect):
        def calc_overlap_ratio(pos_rect):
            x_min = max(pos_rect[0], neg_rect[0])
            y_min = max(pos_rect[1], neg_rect[1])
            x_max = min(pos_rect[2], neg_rect[2])
            y_max = min(pos_rect[3], neg_rect[3])
            if x_max <= x_min or y_max <= y_min:
                return -1
            overlap = (y_max - y_min) * (x_max - x_min)
            return overlap / ((pos_rect[2] - pos_rect[0]) *
                              (pos_rect[3] - pos_rect[1]))
        for pos in labelinfo:
            overlap_ratio = calc_overlap_ratio(pos)
            if g_overlap_ratio <= overlap_ratio:
                return False
        return True

    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    if not os.path.exists(imagepath):
        raise IOError(imagepath + " not found !")
    imagename = imagepath.split("/")[-1].split(".")[0]
    img = cv2.imread(imagepath)
    img_height, img_width = img.shape[:2]
    half_length = 28
    pos_num = len(labelinfo)  #正样本的数量
    neg_num = 10 * pos_num  #在给定的图中寻找负样本的数量
    # TODO(liuguiyangnwpu@163.com): 改进负样本的采样策略
    '''
    通过随机策略产生一堆点，将这些点作为候选框的中心点，确定半宽和半长的大小
    确定这个候选框是否合法，然后是否和正样本的重复率是是否大于给定的一个阈值
    '''
    neg_subimgs = list()
    while len(neg_subimgs) < neg_num:
        c_x = random.randint(int(1.2 * half_length),
                             img_width-int(1.2*half_length))
        c_y = random.randint(int(1.2 * half_length),
                             img_height - int(1.2 * half_length))
        # xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        tmp = [c_x-half_length, c_y-half_length, c_x+half_length, c_y+half_length]
        if check_valid(tmp):
            neg_subimgs.append(tmp)

    # 将负样本保存在文件系统中
    cnt = 0
    for pos in neg_subimgs:
        cnt += 1
        xmin, ymin, xmax, ymax = pos[:]
        subimg = img[ymin:ymax, xmin:xmax]
        if subimg.shape[2] == 3:
            subimg = cv2.cvtColor(subimg, cv2.COLOR_RGB2GRAY)
        savename = imagename + "_" + str(cnt) + ".png"
        cv2.imwrite(savedir + savename, subimg)


def load_annotation():
    if not (os.path.exists(annotation_dir) and os.path.isdir(annotation_dir)):
        raise IOError("%s Not Found !" % annotation_dir)
    annotation_lists = os.listdir(annotation_dir)
    for annotation_file in annotation_lists:
        abs_anno_path = annotation_dir + annotation_file
        anno_targets = extract_airplane_posinfo(abs_anno_path)
        abs_src_path = image_dir + ".".join(
            annotation_file.split(".")[:-1] + ["png"])
        print(abs_anno_path)
        # saveTarget(abs_src_path, anno_targets, target_save_dir)
        negative_sample(abs_src_path, anno_targets, neg_samples_save_dir)
        # break


# 从样本中随机裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def create_ssd_training_samples():
    def _crop_valid(area, anno_targets):
        anno_res = []
        for info in anno_targets:
            if ((info[0] >= area[0] and info[1] >= area[1]) and
                (info[2] <= area[2] and info[3] <= area[3])):
                anno_res.append([info[0] - area[0], info[1] - area[1],
                                 info[2] - area[0], info[3] - area[1]])
            if (info[0] >= area[0] and info[1] >= area[1] and
                info[0] < area[2] and info[1] < area[3] and
                (not (info[2] <= area[2] and info[3] <= area[3]))):
                base = (info[2] - info[0]) * (info[3] - info[1])
                x_max_min = min(info[2], area[2])
                y_max_min = min(info[3], area[3])
                new_square = (x_max_min - info[0]) * (y_max_min - info[1])
                if new_square / base >= 0.75:
                    anno_res.append([info[0] - area[0], info[1] - area[1],
                                     x_max_min - area[0], y_max_min - area[1]])
        return anno_res

    def _random_crop(src_image, anno_targets, save_prefix):
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        while len(crop_list) < 10:
            u_x = random.randint(0, img_width - SSD_IMG_W)
            u_y = random.randint(0, img_height - SSD_IMG_H)
            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        # 保存裁剪出来的结果
        if not os.path.isdir(ssd_sample_dir):
            os.makedirs(ssd_sample_dir)
        if not os.path.isdir(ssd_sample_dir + "src"):
            os.makedirs(ssd_sample_dir + "src")
        if not os.path.isdir(ssd_sample_dir + "annotation"):
            os.makedirs(ssd_sample_dir + "annotation")

        for i in range(len(crop_list)):
            x0, y0, x1, y1 = crop_list[i]
            # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
            sub_img = src_img[y0:y1, x0:x1]
            f_name = save_prefix + "_%d_%d_%d_%d_%d.png" % (x0, y0, x1, y1, i)
            cv2.imwrite(ssd_sample_dir + "src/" + f_name, sub_img)
            f_name = save_prefix + "_%d_%d_%d_%d_%d.json" % (x0, y0, x1, y1, i)
            with codecs.open(
                        ssd_sample_dir + "annotation/" + f_name, "w") as handle:
                handle.write(json.dumps(anno_list[i]))
            print("save %s" % f_name)


    if not (os.path.exists(annotation_dir) and os.path.isdir(annotation_dir)):
        raise IOError("%s Not Found !" % annotation_dir)
    annotation_lists = os.listdir(annotation_dir)
    cnt = 0
    for annotation_file in annotation_lists:
        abs_anno_path = annotation_dir + annotation_file
        anno_targets = extract_airplane_posinfo(abs_anno_path)
        abs_src_path = image_dir + ".".join(
            annotation_file.split(".")[:-1] + ["png"])
        if not os.path.exists(abs_src_path):
            continue
        image_name = ".".join(annotation_file.split(".")[:-1])
        src_img = cv2.imread(abs_src_path)
        _random_crop(src_image=src_img, anno_targets=anno_targets,
                     save_prefix=image_name)
        cnt += 1
        if cnt > 1000:
            break
        # break



if __name__ == "__main__":
    create_ssd_training_samples()