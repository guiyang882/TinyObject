# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
from datetime import datetime

import __init
import json
import cv2
import numpy as np
import tensorflow as tf
from mainmodels.models.tradition.cnn import model
from mainmodels.models.tradition.config import g_CNNConfig
from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.tools.data_prep import calc_iou

from mainmodels.models.ssd.inference_with_tradition import interface_with_cnn

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', g_CNNConfig.eval_log_dir,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', g_CNNConfig.train_log_dir,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 30 * 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples',
                            g_CNNConfig.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('batch_size', g_CNNConfig.batch_size,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def evaluate(ssd_output_batch):
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images = tf.placeholder(
            dtype=tf.float32,
            shape=(None, g_CNNConfig.image_height, g_CNNConfig.image_width,
                   g_CNNConfig.image_depth))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            g_CNNConfig.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = \
                ckpt.model_checkpoint_path.split('/')[-1].split('-')[
                    -1]
            else:
                print('No checkpoint file found')
                return

            val = sess.run(logits, feed_dict={images: ssd_output_batch})
            # print(val)
            print("len of cnn output ",len(val))
            prob = tf.nn.softmax(val)
            # print(prob.eval())
            return prob.eval()

RATIO=[0.8,1,1.2]

def boxes_generation(box):
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    box_h = int(math.fabs(box_h))
    box_w = int(math.fabs(box_w))

    min_side = min(box_h,box_w)
    max_side = max(box_h,box_w)
    center = [round((box[2] + box[0])/2), round((box[3] + box[1])/2)]

    gene_boxes = []
    for cell_ratio in RATIO:
        upper_bound = max(max_side,round(max_side / cell_ratio))

        for new_width in range(min_side, upper_bound):
            new_height = new_width * cell_ratio
            x1,y1,x2,y2 = center[0] - round(new_width / 2),\
                          center[1] - round(new_height / 2),\
                          center[0] + round(new_width / 2),\
                          center[1] + round(new_height / 2)
            if x1 >= x2 or y1 >= y2:
                continue
            if x2 - x1 <= 15 or y2 - y1 <= 15:
                continue
            gene_boxes.append([x1,y1,x2,y2])

    return gene_boxes


def crop_image_with_boxes(filepath, boxes):
    if not os.path.exists(filepath):
        raise IOError("%s not exists !" % filepath)
    image = cv2.imread(filepath, 0)
    new_boxes = []
    crop_subImg_list = []
    for box in boxes:
        x1, y1, x2, y2 = box
        #print(x1, y1, x2, y2)
        subImg = image[y1:y2, x1:x2]
        img_shape = subImg.shape
        if img_shape[0] <= 15 or img_shape[1] <= 15:
            continue
        new_boxes.append(box)
        subImg = cv2.resize(subImg,
                         (g_CNNConfig.image_height, g_CNNConfig.image_width),
                         interpolation=cv2.INTER_LINEAR)
        subImg = subImg.reshape((1, subImg.shape[0], subImg.shape[1], 1))
        crop_subImg_list.append(subImg)

    tf_batch_imgs = crop_subImg_list[0]
    for val in crop_subImg_list[1:]:
        tf_batch_imgs = np.concatenate((tf_batch_imgs, val),
                                   axis=0)
    tf_batch_imgs = tf_batch_imgs / 255.0
    tf_batch_imgs -= 0.5
    tf_batch_imgs *= 2.0
    return tf_batch_imgs,new_boxes

def NMS(boxes,class_label,class_prob):
    if not os.path.exists(g_SSDConfig.tt100k_traffic_sign_path):
        raise IOError("tt100k_traffic_sign_path Not Found !")
    class_boxes = {}
    with open(g_SSDConfig.tt100k_traffic_sign_path, "r") as handle:
        traffic_label_dict = json.load(handle)
        for key, val in traffic_label_dict.items():
            class_boxes[val] = list()

    for i,box_coords in enumerate(boxes):
        cls = class_label[i]
        cls_prob = class_prob[i]
        if cls_prob > g_SSDConfig.CONF_THRESH and cls > 0.:
            box = (*box_coords, cls, cls_prob)
            # print("box",box)
            if len(class_boxes[cls]) == 0:
                class_boxes[cls].append(box)
            else:
                suppressed = False  # did this box suppress other box(es)?
                overlapped = False  # did this box overlap with other box(es)?
                for other_box in class_boxes[cls]:
                    iou = calc_iou(box[:4], other_box[:4])
                    if iou > g_SSDConfig.NMS_IOU_THRESH:
                        overlapped = True
                        # If current box has higher confidence than other box
                        if box[5] > other_box[5]:
                            class_boxes[cls].remove(other_box)
                            suppressed = True
                if suppressed or not overlapped:
                    class_boxes[cls].append(box)

    new_boxes = []
    for cls in class_boxes.keys():
        for class_box in class_boxes[cls]:
            new_boxes.append(class_box)
    new_boxes = np.array(new_boxes)
    # print(new_boxes)
    return new_boxes


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)

    proj_dir = "/Volumes/projects/NWPU-VHR-10-dataset"
    sample_images_dir = "/".join([proj_dir, "demo_test"])
    demo_lists = os.listdir(sample_images_dir)

    input_files = []
    for item in demo_lists:
        if item.endswith("png") or item.endswith("jpg"):
            input_files.append(sample_images_dir+"/"+item)
    # input_files = input_files[:1]
    ret_ssd_res = interface_with_cnn(input_files)
    for file_path, boxes in ret_ssd_res.items():
        print("file_path: %s" % file_path)
        selected_boxes = []
        for box in boxes:
            tmp = boxes_generation(box)
            selected_boxes.extend(tmp)
        # new_selected_box = sample_num * 4
        ssd_output_images, new_selected_boxes = crop_image_with_boxes(file_path, selected_boxes)
        print(ssd_output_images.shape)
        prob_matric = evaluate(ssd_output_images)

        class_label_list = [list(np.where(np.array(prob_line)
                                          == max(prob_line))[0])[0] for prob_line in prob_matric]
        class_prob_list = [prob_matric[i][cls] for i, cls in enumerate(class_label_list)]
        # nms_boxes = sample_num * 6
        nms_boxes = NMS(new_selected_boxes,class_label_list,class_prob_list)

        image = cv2.imread(file_path)
        print("len of selected_boxes: %d" % len(new_selected_boxes))
        for i, selected_box in enumerate(new_selected_boxes):
            box_coords = selected_box
            pred_prob = class_label_list[i]
            pred_cls = class_prob_list[i]
            # print(box_coords)
            try:
                image = cv2.rectangle(image, tuple(box_coords[:2]), tuple(box_coords[2:4]), (0, 255, 0))
            except:
                print(tuple(box_coords[:2]), tuple(box_coords[2:4]))
            label_str = '%d %.2f' % (pred_cls, pred_prob)
            image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
        no_nms_img_filename = "/Volumes/projects/NWPU-VHR-10-dataset/demo_test_res/" + os.path.split(file_path)[1]
        cv2.imwrite(no_nms_img_filename, image)
        print(no_nms_img_filename)
        # break

        image_orig = cv2.imread(file_path)
        print("len of nms_boxes: %d" % len(nms_boxes))
        for i, nms_box in enumerate(nms_boxes):
            box_coords = [int(cell) for cell in nms_box[:5]]
            box_coords.append(nms_box[5])
            print(box_coords)
            try:
                image_orig = cv2.rectangle(image_orig, tuple(box_coords[:2]), tuple(box_coords[2:4]), (0, 255, 0))
            except:
                print("error in box drawing", tuple(box_coords[:2]), tuple(box_coords[2:4]))
            pred_prob = box_coords[5]
            pred_cls = box_coords[4]
            label_str = '%d %.2f' % (pred_cls, pred_prob)
            image_orig = cv2.putText(image_orig, label_str, (box_coords[0], box_coords[1]), 0,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
        nms_img_filename = "/Volumes/projects/NWPU-VHR-10-dataset/demo_test_res/" +\
                                            str(os.path.split(file_path)[1].split('.')[0])\
                                                     +"-nms."+ str(os.path.split(file_path)[1].split('.')[1])
        print(nms_img_filename)
        cv2.imwrite(nms_img_filename, image_orig)
        # print(nms_img_filename)


if __name__ == '__main__':
    tf.app.run()
