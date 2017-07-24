# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import json

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import __init

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.ssdmodel import SSDModel
from mainmodels.models.ssd.tools.NMS import nms


def run_inference(image, model, sess):
    """
	Run inference on a given image

	Arguments:
		* image: Numpy array representing a single RGB image
		* model: Dict of tensor references returned by SSDModel()
		* sess: TensorFlow session reference

	Returns:
		* Numpy array representing annotated image
	"""
    # Save original image in memory
    image = np.array(image)
    image_orig = np.copy(image)

    # Get relevant tensors
    x = model['x']
    is_training = model['is_training']
    preds_conf = model['preds_conf']
    preds_loc = model['preds_loc']
    probs = model['probs']

    # Convert image to PIL Image, resize it, convert to grayscale (if necessary), convert back to numpy array
    image = Image.fromarray(image)
    orig_w, orig_h = image.size
    if g_SSDConfig.NUM_CHANNELS == 1:
        image = image.convert('L')  # 8-bit grayscale
    image = image.resize((g_SSDConfig.IMG_W, g_SSDConfig.IMG_H),
                         Image.LANCZOS)  # high-quality downsampling filter
    image = np.asarray(image)

    images = np.array([image])  # create a "batch" of 1 image
    if g_SSDConfig.NUM_CHANNELS == 1:
        images = np.expand_dims(images,
                                axis=-1)  # need extra dimension of size 1 for grayscale

    # Perform object detection
    t0 = time.time()  # keep track of duration of object detection + NMS
    preds_conf_val, preds_loc_val, probs_val = sess.run(
        [preds_conf, preds_loc, probs],
        feed_dict={x: images, is_training: False})
    print('Inference took %.1f ms (%.2f fps)' % (
        (time.time() - t0) * 1000, 1 / (time.time() - t0)))

    #print("preds_conf_val", preds_conf_val)
    #print("preds_loc_val", preds_loc_val)
    #print("probs_val", probs_val)
    # Gather class predictions and confidence values
    y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
    y_pred_conf = y_pred_conf.astype('float32')
    prob = probs_val[0]

    # Gather localization predictions
    y_pred_loc = preds_loc_val[0]

    # Perform NMS
    boxes = nms(y_pred_conf, y_pred_loc, prob)
    print('Inference + NMS took %.1f ms (%.2f fps)' % (
        (time.time() - t0) * 1000, 1 / (time.time() - t0)))

    # Rescale boxes' coordinates back to original image's dimensions
    # Recall boxes = [[x1, y1, x2, y2, cls, cls_prob], [...], ...]
    scale = np.array(
        [orig_w / g_SSDConfig.IMG_W,
         orig_h / g_SSDConfig.IMG_H,
         orig_w / g_SSDConfig.IMG_W,
         orig_h / g_SSDConfig.IMG_H])
    if len(boxes) > 0:
        boxes[:, :4] = boxes[:, :4] * scale

    #print("boxes: ", boxes)
    # Draw and annotate boxes over original image, and return annotated image
    ret_ssd_res = []
    for box in boxes:
        # Get box parameters
        box_coords = [int(round(x)) for x in box[:4]]
        cls = int(box[4])
        cls_prob = box[5]
        ret_ssd_res.append(box_coords)

    return ret_ssd_res


def generate_output(input_files, options):
    """
	Generate annotated images, videos, or sample images, based on mode
	"""
    # Create output directory 'inference_out/' if needed
    if not os.path.isdir(options.inference_out):
        try:
            os.mkdir(options.inference_out)
        except FileExistsError:
            raise IOError('Error: Cannot mkdir ./inference_out')

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # "Instantiate" neural network, get relevant tensors
        model = SSDModel()

        # Load trained model
        saver = tf.train.Saver()
        print('Restoring previously trained model at %s' %
              g_SSDConfig.PRETRAIN_MODEL_PATH)
        saver.restore(sess, g_SSDConfig.PRETRAIN_MODEL_PATH)

        data2cnn = dict()
        if options.mode == 'image':
            for image_file in input_files:
                print('Running inference on %s' % image_file)
                image_orig = np.asarray(Image.open(image_file))
                ret_ssd_res = run_inference(image_orig, model, sess)
                data2cnn[image_file] = ret_ssd_res

        return data2cnn


def interface_with_cnn(file_path_list):
    sign_file_path = ""
    if g_SSDConfig.MODEL == "AlexNet":
        proj_dir = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/StandardData"
        sign_file_path = g_SSDConfig.tt100k_traffic_sign_path
    elif g_SSDConfig.MODEL == "NWPUNet":
        proj_dir = "/Volumes/projects/NWPU-VHR-10-dataset"
        sign_file_path = g_SSDConfig.nwpu_sign_path
    else:
        raise IOError("Not !!!!")
    print(sign_file_path)
    class RunOption(object):
        input_dir = "input_dir"
        mode = "image"
        sign_file_path =  ""
        inference_out = "/".join([proj_dir, "demo_test_res"])
        sample_images_dir = "/".join([proj_dir, "demo_test"])
        def __init__(self, sign_path):
            self.sign_file_path = sign_path

    options = RunOption(sign_file_path)
    return generate_output(file_path_list, options)

