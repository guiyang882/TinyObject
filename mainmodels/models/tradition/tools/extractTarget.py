# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import shutil
import codecs

import cv2
import numpy as np
import tensorflow as tf
import xml.dom.minidom

# 数据集切分
# srcdir = "/Users/liuguiyang/Downloads/AirplaneSamples/train/JL1ST/"
# savedir = "/Users/liuguiyang/Downloads/AirplaneSamples/test/JL1ST/"
def splitDataSet(srcdir, savedir, ratio=0.2):
    if os.path.isdir(srcdir) == False:
        raise IOError(srcdir + " dir not found !")
    if os.path.isdir(savedir) == False:
        os.makedirs(savedir)
    imagelist = []
    for filename in os.listdir(srcdir):
        if "png" in filename or "jpg" in filename:
            imagelist.append(filename)
    slice = random.sample(imagelist, int(len(imagelist) * ratio))
    for name in slice:
        labelname = name.replace("png", "lif")
        shutil.move(srcdir + name, savedir + name)
        shutil.move(srcdir + labelname, savedir + labelname)
        print(labelname, name)

# 给定一个标记文件，找到对应的目标的位置信息
def extractAirplanePosInfo(filename):
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
    print(imagename)
    img = cv2.imread(imagepath)
    cnt = 0
    for pos in labelinfo:
        cnt += 1
        print(pos)
        xmin, ymin, xmax, ymax = pos[0], pos[1], pos[2], pos[3]
        targetname = pos[4]
        width = ymax - ymin
        height = xmax - xmin
        subimg = img[ymin:ymax, xmin:xmax]
        print(subimg.shape)
        if subimg.shape[2] == 3:
            subimg = cv2.cvtColor(subimg, cv2.COLOR_RGB2GRAY)
        savename = imagename + "_" + str(cnt) + ".png"
        cv2.imwrite(savedir+savename, subimg)

# 提取图像中对应的飞机的位置信息
# srcdir = "/Users/liuguiyang/Downloads/AirplaneSamples/test/JL1ST/"
# savedir = "/Users/liuguiyang/Downloads/AirplaneSamples/corp/JL1ST/"
def fetchTargetPosition(srcdir, savedir, subfix='lif'):
    if os.path.isdir(srcdir) == False:
        raise IOError(srcdir + " dir not found !")
    imagelist = []
    for filename in os.listdir(srcdir):
        if "png" in filename or "jpg" in filename:
            labelname = ".".join(filename.split(".")[0:-1]) + "." + subfix
            imagelist.append((filename, labelname))
    for (filename, labelname) in imagelist:
        print(filename, labelname)
        targetPoses = extractAirplanePosInfo(srcdir+labelname)
        print(targetPoses)
        saveTarget(srcdir+filename, targetPoses, savedir)
        # break

def create_Samples(index_file_path, label, save_name):
    """该函数将对采样出来的样本进行处理，得到固定大小的图像"""
    datalist, labels = None, []
    with codecs.open(index_file_path, 'r', "utf8") as handle:
        for line in handle.readlines():
            line = line.strip()
            if not os.path.exists(line):
                continue
            # print(line)
            image = cv2.imread(line, 0)
            h, w = 56, 56
            if image.shape[0] != h or image.shape[1] != w:
                image = cv2.resize(image, (h, w),
                                   interpolation=cv2.INTER_LINEAR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape((1, image.shape[0], image.shape[1], 1))
            # print(type(image), image.shape)
            if datalist is None:
                datalist = image
            else:
                datalist = np.concatenate((datalist, image), axis=0)
            labels.append(label)
    info = {
        "images": datalist,
        "labels": labels
    }
    with open("/".join(index_file_path.split("/")[:-1]+[save_name]),
              "wb") as handle:
        pickle.dump(info, handle)
    print(datalist.shape, datalist.dtype)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def prepare_Train_Test_Sample(positive_path, negative_path, save_prefix):
    if not os.path.exists(positive_path) or not os.path.exists(negative_path):
        raise IOError("File Not Exists !")
    pos_samples = pickle.load(open(positive_path, "rb"))
    neg_samples = pickle.load(open(negative_path, "rb"))
    pos_idx_list = [i for i in range(len(pos_samples["labels"]))]
    neg_idx_list = [i for i in range(len(neg_samples["labels"]))]
    for i in range(2):
        random.shuffle(pos_idx_list)
        random.shuffle(neg_idx_list)
    ratio = 0.8
    pos_train_len = int(len(pos_idx_list) * ratio)
    neg_train_len = int(len(neg_idx_list) * ratio)
    pos_train_idx = random.sample(pos_idx_list, pos_train_len)
    neg_train_idx = random.sample(neg_idx_list, neg_train_len)

    pos_test_idx, neg_test_idx = [], []
    for i in pos_idx_list:
        if i not in pos_train_idx:
            pos_test_idx.append(i)
    for i in neg_idx_list:
        if i not in neg_train_idx:
            neg_test_idx.append(i)
    print(len(pos_train_idx), len(neg_train_idx))
    print(len(pos_test_idx), len(neg_test_idx))

    def __save(_sample_data, _sample_idx):
        t_data = _sample_data["images"]
        t_label = _sample_data["labels"]
        res_data = []
        h, w = 56, 56
        for idx in _sample_idx:
            t = t_data[idx].reshape(1, h*w*1).tolist()[0]
            t.append(t_label[idx])
            res_data.append(t)
        res_data = np.array(res_data, dtype=np.uint8)
        print(res_data.shape)
        return res_data

    p_train = __save(pos_samples, pos_train_idx)
    p_test = __save(pos_samples, pos_test_idx)
    n_train = __save(neg_samples, neg_train_idx)
    n_test = __save(neg_samples, neg_test_idx)
    final_train_sample = np.concatenate((p_train, n_train))
    final_test_sample = np.concatenate((p_test, n_test))
    np.random.shuffle(final_train_sample)
    np.random.shuffle(final_train_sample)
    np.random.shuffle(final_test_sample)
    np.random.shuffle(final_test_sample)
    print(final_train_sample.shape)
    print(final_test_sample.shape)

    def __save_tfrecords(file_path, np_samples):
        print("save to " + file_path)
        if os.path.exists(file_path):
            os.remove(file_path)
        writer = tf.python_io.TFRecordWriter(file_path)
        for idx in range(len(np_samples)):
            feature = np_samples[idx]
            # print(feature.shape, feature.dtype)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'feature': _bytes_feature(feature.tobytes())
                    # "label": _bytes_feature(label.to_bytes())
                }))
            writer.write(example.SerializeToString())
        writer.close()

    file_path = "/".join([save_prefix, "train.tfrecords"])
    __save_tfrecords(file_path, final_train_sample)
    file_path = "/".join([save_prefix, "test.tfrecords"])
    __save_tfrecords(file_path, final_test_sample)

def prepare_rpn_list(save_train_path, save_test_path):
    """用来准备RPN网络训练和测试的数据index文件"""
    train_dir_name = "/Users/liuguiyang/Downloads/AirplaneSamples/Positive" \
                   "/train/JL1ST"
    test_dir_name = "/Users/liuguiyang/Downloads/AirplaneSamples/Positive/test"
    if not os.path.exists(train_dir_name) or not os.path.exists(test_dir_name):
        raise IOError("prepare_rpn_list file path not found !")

    def _prepare_list(dir_name):
        res_list = []
        file_list = os.listdir(train_dir_name)
        for name in file_list:
            if name.endswith("png"):
                label_name = None
                if name.replace("png", "lif") in file_list:
                    label_name = name.replace("png", "lif")
                elif name.replace("png", "xml") in file_list:
                    label_name = name.replace("png", "xml")
                else:
                    raise IOError(
                        "In Train File Dir, no such " + name + "label")
                res_list.append([name, label_name])
        return res_list

    # prepare the train file list
    train_list = _prepare_list(train_dir_name)
    with open(save_train_path, "w") as handle:
        for item in train_list:
            handle.write("/".join([train_dir_name, item[0]]))
            handle.write(",")
            handle.write("/".join([train_dir_name, item[1]]))
            handle.write("\n")

    # prepare the test file list
    test_list = _prepare_list(test_dir_name)
    with open(save_test_path, "w") as handle:
        for item in test_list:
            handle.write("/".join([test_dir_name, item[0]]))
            handle.write(",")
            handle.write("/".join([test_dir_name, item[1]]))
            handle.write("\n")


if __name__ == "__main__":
    index_file_path = u"/Volumes/projects/第三方数据下载/JL1ST/index_pos.txt"
    label = 1
    save_name = "positive.pkl"
    create_Samples(index_file_path, label, save_name)

    index_file_path = u"/Volumes/projects/第三方数据下载/JL1ST/index_neg.txt"
    label = 0
    save_name = "negative.pkl"
    create_Samples(index_file_path, label, save_name)

    pos_path = "/Volumes/projects/第三方数据下载/JL1ST/positive.pkl"
    neg_path = "/Volumes/projects/第三方数据下载/JL1ST/negative.pkl"
    save_prefix = "/Volumes/projects/第三方数据下载/JL1ST/"
    prepare_Train_Test_Sample(pos_path, neg_path, save_prefix)