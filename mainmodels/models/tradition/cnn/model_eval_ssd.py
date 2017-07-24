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

import cv2
import numpy as np
import tensorflow as tf
from mainmodels.models.tradition.cnn import model
from mainmodels.models.tradition.config import g_CNNConfig

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
            dtype=tf.uint8,
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
            print(val)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)


RATIO=[0.8,1,1.2]

def boxes_generation(box):
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]

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
    crop_subImg_list = []
    for box in boxes:
        x1, y1, x2, y2 = box
        #print(x1, y1, x2, y2)
        subImg = image[y1:y2, x1:x2]
        subImg = cv2.resize(subImg,
                         (g_CNNConfig.image_height, g_CNNConfig.image_width),
                         interpolation=cv2.INTER_LINEAR)
        subImg = subImg.reshape((1, subImg.shape[0], subImg.shape[1], 1))
        crop_subImg_list.append(subImg)

    tf_batch_imgs = crop_subImg_list[0]
    for val in crop_subImg_list[1:]:
        tf_batch_imgs = np.concatenate((tf_batch_imgs, val),
                                   axis=0)
    return tf_batch_imgs

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

    ret_ssd_res = interface_with_cnn(input_files)
    for file_path, boxes in ret_ssd_res.items():
        selected_boxes = []
        for box in boxes:
            tmp = boxes_generation(box)
            selected_boxes.extend(tmp)
        ssd_output_images = crop_image_with_boxes(file_path, selected_boxes)
        print(ssd_output_images.shape)
        evaluate(ssd_output_images)
        break


if __name__ == '__main__':
    tf.app.run()
