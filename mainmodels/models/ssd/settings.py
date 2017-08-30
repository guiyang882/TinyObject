# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-4])
print(proj_dir)

class SSDConfig(object):
    # Model selection and dependent parameters
    MODEL = 'JLFirst'  # AlexNet/VGG16/ResNet50

    # Default boxes
    # DEFAULT_BOXES = ((x1_offset, y1_offset, x2_offset, y2_offset), (...), ...)
    # Offset is relative to upper-left-corner and lower-right-corner of the feature map cell
    if MODEL == "AlexNet":
        DEFAULT_BOXES = (
            (-0.5, -0.5, 0.5, 0.5), (0.2, 0.2, -0.2, -0.2),
            (-0.8, -0.2, 0.8, 0.2), (-0.2, -0.8, 0.2, 0.8)
        )
    elif MODEL == "NWPUNet":
        DEFAULT_BOXES = (
            (-0.5, -0.5, 0.5, 0.5), (0.2, 0.2, -0.2, -0.2),
            (-0.8, -0.2, 0.8, 0.2), (-0.2, -0.8, 0.2, 0.8)
        )
    elif MODEL == "JLFirst":
        DEFAULT_BOXES = (
            (-0.95, -0.95, 0.95, 0.95),
            (-0.75, -0.75, 0.75, 0.75),
            (-0.55, -0.55, 0.55, 0.55),
            (-0.75, -0.65, 0.75, 0.65),
            (-0.65, -0.75, 0.65, 0.75),
            (-0.95, -0.65, 0.95, 0.65),
            (-0.65, -0.95, 0.65, 0.95),
            (-1.15, -0.95, 0.75, 0.95),
            (-0.95, -0.75, 0.55, 0.75),
            (-0.75, -0.55, 0.35, 0.55),
            (-0.95, -0.65, 0.55, 0.65),
            (-0.85, -0.75, 0.45, 0.75),
            (-1.15, -0.65, 0.75, 0.65),
            (-0.85, -0.95, 0.45, 0.95),
            (-0.75, -0.95, 1.15, 0.95),
            (-0.55, -0.75, 0.95, 0.75),
            (-0.35, -0.55, 0.75, 0.55),
            (-0.55, -0.65, 0.95, 0.65),
            (-0.45, -0.75, 0.85, 0.75),
            (-0.75, -0.65, 1.15, 0.65),
            (-0.45, -0.95, 0.85, 0.95),
            (-0.95, -1.15, 0.95, 1.15),
            (-0.75, -0.95, 0.75, 0.95),
            (-0.55, -0.75, 0.55, 0.75),
            (-0.75, -0.85, 0.75, 0.85),
            (-0.65, -0.95, 0.65, 0.95),
            (-0.95, -0.85, 0.95, 0.85),
            (-0.65, -1.15, 0.65, 1.15),
            (-0.95, -0.75, 0.95, 1.15),
            (-0.75, -0.55, 0.75, 0.95),
            (-0.55, -0.35, 0.55, 0.75),
            (-0.75, -0.45, 0.75, 0.85),
            (-0.65, -0.55, 0.65, 0.95),
            (-0.95, -0.45, 0.95, 0.85),
            (-0.65, -0.75, 0.65, 1.15)
        )
    else:
        pass
    NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)

    # Constants (TODO: Keep this updated as we go along)
    if MODEL == "AlexNet":
        NUM_CLASSES = 222  # 221 signs + 1 background class
    elif MODEL == "NWPUNet":
        NUM_CLASSES = 6  # 8 signs + 1 background class
    elif MODEL == "JLFirst":
        NUM_CLASSES = 2  # 1 airplane + 1 background class
    else:
        raise NotImplementedError('Model not implemented')
    NUM_CHANNELS = 3  # grayscale->1, RGB->3
    NUM_PRED_CONF = NUM_DEFAULT_BOXES * NUM_CLASSES  # number of class predictions per feature map cell
    NUM_PRED_LOC = NUM_DEFAULT_BOXES * 4  # number of localization regression predictions per feature map cell

    # Bounding box parameters
    IOU_THRESH = 0.50  # match ground-truth box to default boxes exceeding
    # this IOU threshold, during data prep
    NMS_IOU_THRESH = 0.2  # IOU threshold for non-max suppression

    # Negatives-to-positives ratio used to filter training data
    NEG_POS_RATIO = 5  # negative:positive = NEG_POS_RATIO:1

    # Class confidence threshold to count as detection
    CONF_THRESH = 0.62

    if MODEL == 'AlexNet':
        IMG_H, IMG_W = 260, 400
        # feature map sizes for SSD hooks via TensorBoard visualization (HxW)
        FM_SIZES = [[31, 48], [15, 23], [8, 12], [4, 6]]
    if MODEL == "NWPUNet":
        IMG_H, IMG_W = 400, 600
        # feature map sizes for SSD hooks via TensorBoard visualization (HxW)
        # FM_SIZES = [[100, 150], [50, 75], [25, 38], [13, 19]]
        FM_SIZES = [[50, 75], [25, 37], [13, 19]]
    if MODEL == "JLFirst":
        IMG_H, IMG_W = 512, 512
        FM_SIZES = [[32, 32], [16, 16], [8, 8]]
    else:
        raise NotImplementedError('Model not implemented')

    # Model hyper-parameters
    OPT = tf.train.AdadeltaOptimizer()
    REG_SCALE = 1e-2  # L2 regularization strength
    LOC_LOSS_WEIGHT = 1.  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss

    # for Multi-GPU learning
    GPU_NUMS = 1
    TOWER_NAME = "TOWER"
    INIT_LEARNING_RATE = 0.001
    NUM_EPOCHS_PER_DECAY = 350.0
    LEARNING_RATE_DECAY_FACTOR = 0.1
    MOVING_AVERAGE_DECAY = 0.9999

    # Training process
    RESUME = False  # resume training from previously saved model?
    NUM_EPOCH = 5000
    BATCH_SIZE = 16  # batch size for training (relatively small)
    VALIDATION_SIZE = 0.05  # fraction of total training set to use as validation set
    SAVE_MODEL = True  # save trained model to disk?

    if MODEL == "AlexNet":
        MODEL_SAVE_PATH = "/".join([proj_dir, "mainmodels", "log", "ssd",
                                    "train", "model.ckpt"])
        LOSS_HISTORY_PATH = "/".join([proj_dir, "mainmodels", "log", "ssd",
                                      "loss_history.pkl"])
        TENSORBOARD_SAVE_PATH = MODEL_SAVE_PATH
        DATASET_BASE_DIR = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/StandardData"
    elif MODEL == "NWPUNet":
        MODEL_SAVE_PATH = "/".join([proj_dir, "mainmodels", "log", "nwpu",
                                    "train", "model.ckpt"])
        LOSS_HISTORY_PATH = "/".join([proj_dir, "mainmodels", "log", "nwpu",
                                      "loss_history.pkl"])
        TENSORBOARD_SAVE_PATH = MODEL_SAVE_PATH
        DATASET_BASE_DIR = "/Volumes/projects/NWPU-VHR-10-dataset"
    elif MODEL == "JLFirst":
        DATASET_BASE_DIR = "/home/ai-i-liuguiyang/repos_ssd/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD"
        MODEL_SAVE_PATH = DATASET_BASE_DIR + "/model/model.ckpt"
        LOSS_HISTORY_PATH = DATASET_BASE_DIR + "/loss_history.pkl"
        TENSORBOARD_SAVE_PATH = MODEL_SAVE_PATH
    else:
        raise NotImplementedError('Model not implemented')

    PRETRAIN_MODEL_PATH = ""
    TRAIN_DATA_PRE_PATH = "/".join(
        [DATASET_BASE_DIR, "ssd_prepare", "train_data_prep.pkl"])
    TEST_DATA_PRE_PATH = "/".join(
        [DATASET_BASE_DIR, "ssd_prepare", "test_data_prep.pkl"])

    TRAIN_DATA_SRC_DIR = "/".join([DATASET_BASE_DIR, "ssd_src"])
    TEST_DATA_SRC_DIR = TRAIN_DATA_SRC_DIR
    RESIZED_IMAGES_DIR = TRAIN_DATA_SRC_DIR
    label_sign_path = "/".join([DATASET_BASE_DIR, "target.json"])

g_SSDConfig = SSDConfig()
