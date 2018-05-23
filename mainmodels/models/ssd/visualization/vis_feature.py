# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/9/1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.ssdmodel import SSDModel
from mainmodels.models.ssd.tools.NMS_vis import nms_vis
from mainmodels.models.ssd.tools.NMS_vis import check_box_legal
from mainmodels.dataset.show_target import show_annotation_image_file
from mainmodels.dataset.show_target import show_ssd_prepare_boxes

IMG_W, IMG_H = g_SSDConfig.IMG_W, g_SSDConfig.IMG_H
CONF_THRESH = g_SSDConfig.CONF_THRESH
sign_map = {
    0: "bg",
    1: "plane"
}

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 100, 200)]

def read_image_sample(image_abs_file):
    image = Image.open(image_abs_file)
    if g_SSDConfig.NUM_CHANNELS == 1:
        image = image.convert('L')
    image = image.resize((g_SSDConfig.IMG_W, g_SSDConfig.IMG_H), Image.LANCZOS)
    image = np.asarray(image)

    images = np.array([image])  # create a "batch" of 1 image
    if g_SSDConfig.NUM_CHANNELS == 1:
        images = np.expand_dims(images, axis=-1)
    images = images / 127.5 - 1.
    return images

# 根据输入的原始尺寸图像和对应的feature map对应的大小
# 绘制出对应的框在原始图中的框
def vis_one_feature_map(src_image, y_pred_conf, y_pred_loc, prob):
    y_idx = 0
    for idx, fm_size in enumerate(g_SSDConfig.FM_SIZES):
        color = color_list[idx]
        fm_h, fm_w = fm_size  # feature map height and width
        for row in range(fm_h):
            for col in range(fm_w):
                for db in g_SSDConfig.DEFAULT_BOXES:
                    if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
                        xc, yc = row+0.5, col+0.5
                        center_coords = np.array([xc, yc, xc, yc])
                        # predictions are offsets to center of fm cell
                        abs_box_coords = center_coords + y_pred_loc[y_idx * 4: y_idx * 4 + 4]
                        # Calculate predicted box coordinates in actual image
                        scale = np.array(
                            [IMG_W / fm_w, IMG_H / fm_h,
                             IMG_W / fm_w, IMG_H / fm_h])
                        box_coords = abs_box_coords * scale
                        box_coords = [int(round(x)) for x in box_coords]
                        if check_box_legal(box_coords):
                            cls = y_pred_conf[y_idx]
                            cls_prob = prob[y_idx]
                            box = (*box_coords, cls, cls_prob)
                            src_image = cv2.rectangle(
                                src_image,
                                tuple(box_coords[:2]), tuple(box_coords[2:]),
                                color, 2)
                    y_idx += 1
        cv2.imshow("fm_boxes", src_image)
        cv2.waitKey()


def run_inference(images, model, sess):
    # Get relevant tensors
    x = model['x']
    is_training = model['is_training']
    preds_conf = model['preds_conf']
    preds_loc = model['preds_loc']
    probs = model['probs']

    # Perform object detection
    t0 = time.time()
    preds_conf_val, preds_loc_val, probs_val = sess.run(
        [preds_conf, preds_loc, probs],
        feed_dict={x: images, is_training: False})
    print('Inference took %.1f ms (%.2f fps)' % (
        (time.time() - t0) * 1000, 1 / (time.time() - t0)))

    # Gather class predictions and confidence values
    y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
    y_pred_conf = y_pred_conf.astype('float32')
    prob = probs_val[0]
    y_pred_loc = preds_loc_val[0]

    vis_one_feature_map(images[0], y_pred_conf, y_pred_loc, prob)
    print('Inference + NMS took %.1f ms (%.2f fps)' % (
        (time.time() - t0) * 1000, 1 / (time.time() - t0)))


def show_details_by_name(images, model, sess, tensor_name, tensor_shape):
    x = model['x']
    is_training = model['is_training']
    show_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    tensor_values = sess.run([show_tensor],
                             feed_dict={x: images, is_training: False})
    plt.figure(num=tensor_name)
    for tensor_value in tensor_values:
        print(tensor_value.shape)
        n_channel = tensor_value.shape[-1]
        for idx in range(0, n_channel):
            tmp = tensor_value[:, :, :, idx]
            tmp = np.squeeze(tmp, axis=0)
            # n = 1 if (idx+1) % 17 == 0 else (idx+1) % 17
            plt.subplot(tensor_shape[0], tensor_shape[1], idx+1)
            plt.imshow(tmp)
            plt.axis('off')
    plt.show()


def generate_output(image_file):
    images = read_image_sample(image_file)

    # Launch the graph
    with tf.Session() as sess:
        # "Instantiate" neural network, get relevant tensors
        model = SSDModel()

        # Load trained model
        saver = tf.train.Saver()
        print('Restoring trained model at %s' % g_SSDConfig.PRETRAIN_MODEL_PATH)
        saver.restore(sess, g_SSDConfig.PRETRAIN_MODEL_PATH)
        # 主要负责各层卷积模版的提取和可视化
        show_tensor_list = [
            ("conv1/convolution:0", (8, 8)),
            ("conv1/Relu:0", (8, 8)),
            ("pool1/MaxPool:0", (8, 8)),
            ("conv2/convolution:0", (14, 14)),
            ("conv2/Relu:0", (14, 14)),
            ("pool2/MaxPool:0", (14, 14)),
            ("conv3/convolution:0", (20, 20)),
            ("conv3/Relu:0", (20, 20)),
            ("conv4/convolution:0", (20, 20)),
            ("conv4/Relu:0", (20, 20)),
            ("conv5/convolution:0", (20, 20)),
            ("conv5/Relu:0", (20, 20)),
        ]
        # graph_def = tf.get_default_graph().as_graph_def()
        # for node in graph_def.node:
        #     print(node.name)
        show_idx = 2
        show_details_by_name(images, model, sess,
                             show_tensor_list[show_idx][0],
                             show_tensor_list[show_idx][1])
        # run_inference(images, model, sess)


if __name__ == '__main__':
    dir_prefix = "/Volumes/projects/第三方数据下载/JL1ST/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD_AlexNet/test/src"
    input_file_name = "0_1536_0_2048_0604_1271_977_1671_1237_8.png"
    show_annotation_image_file(input_file_name)
    show_ssd_prepare_boxes(input_file_name)
    abs_file_path = dir_prefix + "/" + input_file_name
    generate_output(abs_file_path)