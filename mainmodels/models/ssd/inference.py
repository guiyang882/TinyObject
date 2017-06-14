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

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.ssdmodel import SSDModel
from mainmodels.models.ssd.tools.NMS import nms


def run_inference(image, model, sess, sign_map):
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

    # Draw and annotate boxes over original image, and return annotated image
    image = image_orig
    for box in boxes:
        # Get box parameters
        box_coords = [int(round(x)) for x in box[:4]]
        cls = int(box[4])
        cls_prob = box[5]

        # Annotate image
        image = cv2.rectangle(image, tuple(box_coords[:2]),
                              tuple(box_coords[2:]), (0, 255, 0))
        label_str = '%s %.2f' % (sign_map[cls], cls_prob)
        image = cv2.putText(image, label_str, (box_coords[0], box_coords[1]), 0,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


def generate_output(input_files, options):
    """
	Generate annotated images, videos, or sample images, based on mode
	"""
    if not os.path.exists(options.sign_file_path):
        raise IOError(options.sign_file_path + " not found !")
    # First, load mapping from integer class ID to sign name string
    sign_map = {}
    with open(options.sign_file_path, 'r') as f:
        for line in f:
            sign_id, sign_name = line.strip().split(',')
            sign_map[int(sign_id)] = sign_name
    sign_map[0] = 'background'  # class ID 0 reserved for background class

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
              g_SSDConfig.MODEL_SAVE_PATH)
        saver.restore(sess, g_SSDConfig.MODEL_SAVE_PATH)

        if options.mode == 'image':
            for image_file in input_files:
                print('Running inference on %s' % image_file)
                image_orig = np.asarray(Image.open(image_file))
                image = run_inference(image_orig, model, sess, sign_map)

                head, tail = os.path.split(image_file)
                plt.imsave('./inference_out/%s' % tail, image)
            print('Output saved in inference_out/')

        elif options.mode == 'demo':
            print('Demo mode: Running inference on images in sample_images/')
            image_files = os.listdir(options.sample_images_dir)

            for image_file in image_files:
                print('Running inference on %s/%s' % (
                    options.sample_images_dir, image_file))
                image_orig = np.asarray(
                    Image.open("/".join([options.sample_images_dir, image_file])))
                image = run_inference(image_orig, model, sess, sign_map)
                plt.imshow(image)
                plt.show()


if __name__ == '__main__':
    proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-4])

    class RunOption(object):
        input_dir = "input_dir"
        mode = "demo"
        sign_file_path = "signnames.csv"
        inference_out = "/".join([proj_dir, "data", "inference_out"])
        sample_images_dir = "/".join([proj_dir, "data", "traffic_sign_samples"])

    options = RunOption()
    if options.mode not in ["image", "demo"]:
        raise ValueError('Invalid mode: %s' % options.mode)

    input_files = []
    generate_output(input_files, options)
