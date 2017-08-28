# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import pickle

import __init

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

from mainmodels.models.ssd.settings import g_SSDConfig
from mainmodels.models.ssd.ssdmodel import SSDModel


def next_batch(X, y_conf, y_loc, batch_size):
    """
	Next batch generator
	Arguments:
		* X: List of image file names
		* y_conf: List of ground-truth vectors for class labels
		* y_loc: List of ground-truth vectors for localization
		* batch_size: Batch size

	Yields:
		* images: Batch numpy array representation of batch of images
		* y_true_conf: Batch numpy array of ground-truth class labels
		* y_true_loc: Batch numpy array of ground-truth localization
		* conf_loss_mask: Loss mask for confidence loss, to set NEG_POS_RATIO
	"""
    start_idx = 0
    while True:
        image_files = X[start_idx: start_idx + batch_size]
        y_true_conf = np.array(y_conf[start_idx: start_idx + batch_size])
        y_true_loc = np.array(y_loc[start_idx: start_idx + batch_size])

        # Read images from image_files
        images = []
        for image_file in image_files:
            image_abs_file = "/".join(
                [g_SSDConfig.DATASET_BASE_DIR, image_file])
            image_abs_file = image_file
            image = Image.open(image_abs_file)
            image = np.asarray(image)
            images.append(image)

        images = np.array(images, dtype='float32')

        # Grayscale images have array shape (H, W), but we want shape (H, W, 1)
        if g_SSDConfig.NUM_CHANNELS == 1:
            images = np.expand_dims(images, axis=-1)

        # Normalize pixel values (scale them between -1 and 1)
        images = images / 127.5 - 1.

        # For y_true_conf, calculate how many negative examples we need to satisfy NEG_POS_RATIO
        num_pos = np.where(y_true_conf > 0)[0].shape[0]
        num_neg = g_SSDConfig.NEG_POS_RATIO * num_pos
        y_true_conf_size = np.sum(y_true_conf.shape)

        # Create confidence loss mask to satisfy NEG_POS_RATIO
        if num_pos + num_neg < y_true_conf_size:
            conf_loss_mask = np.copy(y_true_conf)
            conf_loss_mask[np.where(conf_loss_mask > 0)] = 1.

            # Find all (i,j) tuples where y_true_conf[i][j]==0
            zero_indices = np.where(
                conf_loss_mask == 0.)  # ([i1, i2, ...], [j1, j2, ...])
            zero_indices = np.transpose(
                zero_indices)  # [[i1, j1], [i2, j2], ...]

            # Randomly choose num_neg rows from zero_indices, w/o replacement
            chosen_zero_indices = zero_indices[
                np.random.choice(zero_indices.shape[0], int(num_neg), False)]

            # "Enable" chosen negative examples, specified by chosen_zero_indices
            for zero_idx in chosen_zero_indices:
                i, j = zero_idx
                conf_loss_mask[i][j] = 1.
        else:
            # If we have so many positive examples such that num_pos+num_neg >= y_true_conf_size,
            # no need to prune negative data
            conf_loss_mask = np.ones_like(y_true_conf)

        yield (images, y_true_conf, y_true_loc, conf_loss_mask)

        # Update start index for the next batch
        start_idx += batch_size
        if start_idx >= X.shape[0]:
            start_idx = 0


def run_training():
    # Load training and test data
    with open(g_SSDConfig.TRAIN_DATA_PRE_PATH, mode='rb') as f:
        train = pickle.load(f)

    # Format the data
    X_train = []
    y_train_conf = []
    y_train_loc = []
    for image_file in train.keys():
        X_train.append(image_file)
        y_train_conf.append(train[image_file]['y_true_conf'])
        y_train_loc.append(train[image_file]['y_true_loc'])
    X_train = np.array(X_train)
    y_train_conf = np.array(y_train_conf)
    y_train_loc = np.array(y_train_loc)

    # Train/validation split
    X_train, X_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc = \
        train_test_split(
            X_train, y_train_conf, y_train_loc,
            test_size=g_SSDConfig.VALIDATION_SIZE, random_state=1)

    # Launch the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # "Instantiate" neural network, get relevant tensors
        model = SSDModel()
        x = model['x']
        y_true_conf = model['y_true_conf']
        y_true_loc = model['y_true_loc']
        conf_loss_mask = model['conf_loss_mask']
        is_training = model['is_training']
        optimizer = model['optimizer']
        reported_loss = model['loss']

        # Training process
        # TF saver to save/restore trained model
        saver = tf.train.Saver()

        if g_SSDConfig.RESUME:
            print('Restoring previously trained model at %s' %
                  g_SSDConfig.MODEL_SAVE_PATH)
            saver.restore(sess, g_SSDConfig.MODEL_SAVE_PATH)

            # Restore previous loss history
            with open(g_SSDConfig.LOSS_HISTORY_PATH, 'rb') as f:
                loss_history = pickle.load(f)
        else:
            print('Training model from scratch')
            # Variable initialization
            sess.run(tf.global_variables_initializer())

            # For book-keeping, keep track of training and validation loss over epochs, like such:
            # [(train_acc_epoch1, valid_acc_epoch1), (train_acc_epoch2, valid_acc_epoch2), ...]
            loss_history = []

        # Record time elapsed for performance check
        last_time = time.time()
        train_start_time = time.time()

        # Run NUM_EPOCH epochs of training
        for epoch in range(g_SSDConfig.NUM_EPOCH):
            train_gen = next_batch(X_train, y_train_conf, y_train_loc,
                                   g_SSDConfig.BATCH_SIZE)
            num_batches_train = math.ceil(X_train.shape[0] /
                                          g_SSDConfig.BATCH_SIZE)
            losses = []  # list of loss values for book-keeping

            # Run training on each batch
            for _ in range(num_batches_train):
                # Obtain the training data and labels from generator
                images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(
                    train_gen)

                # Perform gradient update (i.e. training step) on current batch
                _, loss = sess.run([optimizer, reported_loss], feed_dict={
                    # _, loss, loc_loss_dbg, loc_loss_mask, loc_loss = sess.run([optimizer, reported_loss, model['loc_loss_dbg'], model['loc_loss_mask'], model['loc_loss']],feed_dict={  # DEBUG
                    x: images,
                    y_true_conf: y_true_conf_gen,
                    y_true_loc: y_true_loc_gen,
                    conf_loss_mask: conf_loss_mask_gen,
                    is_training: True
                })

                losses.append(loss)  # TODO: Need mAP metric instead of raw loss

            # A rough estimate of loss for this epoch (overweights the last batch)
            train_loss = np.mean(losses)
            print("train_loss is: %f" % train_loss)

            # Calculate validation loss at the end of the epoch
            valid_gen = next_batch(X_valid, y_valid_conf, y_valid_loc,
                                   g_SSDConfig.BATCH_SIZE)
            num_batches_valid = math.ceil(X_valid.shape[0] /
                                          g_SSDConfig.BATCH_SIZE)
            losses = []
            for _ in range(num_batches_valid):
                images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(
                    valid_gen)

                # Perform forward pass and calculate loss
                loss = sess.run(reported_loss, feed_dict={
                    x: images,
                    y_true_conf: y_true_conf_gen,
                    y_true_loc: y_true_loc_gen,
                    conf_loss_mask: conf_loss_mask_gen,
                    is_training: False
                })
                losses.append(loss)
            valid_loss = np.mean(losses)
            print("valid_loss is: %f" % valid_loss)

            # Record and report train/validation/test losses for this epoch
            loss_history.append((train_loss, valid_loss))

            # Print accuracy every epoch
            print(
                'Epoch %d -- Train loss: %.4f, Validation loss: %.4f, Elapsed time: %.2f sec' % \
                (epoch + 1, train_loss, valid_loss, time.time() - last_time))
            last_time = time.time()

            if g_SSDConfig.SAVE_MODEL and epoch % 5 == 0:
                # Save model to disk
                save_path = saver.save(sess, g_SSDConfig.MODEL_SAVE_PATH, global_step=epoch)
                print('Trained model saved at: %s' % save_path)

                # Also save accuracy history
                print('Loss history saved at loss_history.pkl')
                with open(g_SSDConfig.LOSS_HISTORY_PATH, 'wb') as f:
                    pickle.dump(loss_history, f)

        total_time = time.time() - train_start_time
        print('Total elapsed time: %d min %d sec' % (
            total_time / 60, total_time % 60))

    # Return final test accuracy and accuracy_history
    return loss_history


if __name__ == '__main__':
    # if g_SSDConfig.MODEL == "AlexNet":
    #     base_dir = "/Volumes/projects/TrafficSign/Tencent-Tsinghua/StandardData" \
    #            "/raw_prep/prep_data"
    # elif g_SSDConfig.MODEL == "NWPUNet":
    #     base_dir = "/Volumes/projects/NWPU-VHR-10-dataset/raw_prep/prep_data"
    # else:
    #     raise NotImplementedError('Model not implemented')

    # prep_train = dict()
    # prep_test = dict()
    # for file_path in os.listdir(base_dir):
    #     if "train" in file_path:
    #         with open(base_dir+"/"+file_path, "rb") as handle:
    #             part_train = pickle.load(handle)
    #             for key, val in part_train.items():
    #                 prep_train[key] = val
    #                 # print(type(val))
    #     if "test" in file_path:
    #         with open(base_dir+"/"+file_path, "rb") as handle:
    #             part_test = pickle.load(handle)
    #             for key, val in part_test.items():
    #                 prep_test[key] = val
    # print(prep_train.keys())
    # with open(g_SSDConfig.TRAIN_DATA_PRE_PATH, 'wb') as save_handle:
    #     pickle.dump(prep_train, save_handle)
    # print(prep_test.keys())
    # with open(g_SSDConfig.TEST_DATA_PRE_PATH, "wb") as handle:
    #     pickle.dump(prep_test, handle)

    run_training()