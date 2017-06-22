# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import pickle

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

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


def perpare_dataset():
    # Load training and test data
    with open(g_SSDConfig.TRAIN_DATA_PRE_PATH, mode='rb') as f:
        train = pickle.load(f)
    # with open('test.p', mode='rb') as f:
    #	test = pickle.load(f)

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
    return X_train, X_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc


model_ph_tensor_on_gpu = dict()

def tower_loss(scope, gpu_id):
    # "Instantiate" neural network, get relevant tensors
    model = SSDModel()
    model_ph_tensor_on_gpu[gpu_id] = model

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

      Note that this function provides a synchronization point across all towers.

      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def run_epoch_on_train_sets(sess, optimizer, reported_loss, X_train,
                            y_train_conf, y_train_loc, gpu_id):
    train_gen = next_batch(
        X_train, y_train_conf, y_train_loc, g_SSDConfig.BATCH_SIZE)
    num_batches_train = math.ceil(
        X_train.shape[0] / g_SSDConfig.BATCH_SIZE)

    # optimizer = model_ph_tensor_on_gpu[gpu_id]["optimizer"]
    # reported_loss = model_ph_tensor_on_gpu[gpu_id]["loss"]
    x = model_ph_tensor_on_gpu[gpu_id]["x"]
    y_true_conf = model_ph_tensor_on_gpu[gpu_id]["y_true_conf"]
    y_true_loc = model_ph_tensor_on_gpu[gpu_id]["y_true_loc"]
    conf_loss_mask = model_ph_tensor_on_gpu[gpu_id]["conf_loss_mask"]
    is_training = model_ph_tensor_on_gpu[gpu_id]["is_training"]

    # Run training on each batch
    losses = []  # list of loss values for book-keeping
    for _ in range(num_batches_train):
        # Obtain the training data and labels from generator
        images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = \
            next(train_gen)

        # Perform gradient update (i.e. training step) on current batch
        _, loss = sess.run([optimizer, reported_loss], feed_dict={
            x: images,
            y_true_conf: y_true_conf_gen,
            y_true_loc: y_true_loc_gen,
            conf_loss_mask: conf_loss_mask_gen,
            is_training: True
        })
        # TODO: Need mAP metric instead of raw loss
        losses.append(loss)

    # A rough estimate of loss for this epoch (overweights the last batch)
    train_loss = np.mean(losses)
    return train_loss

def run_epoch_on_valid_sets(sess, reported_loss, X_valid,
                            y_valid_conf,
                            y_valid_loc, gpu_id):
    # Calculate validation loss at the end of the epoch
    losses = []
    valid_gen = next_batch(
        X_valid, y_valid_conf, y_valid_loc, g_SSDConfig.BATCH_SIZE)
    num_batches_valid = math.ceil(
        X_valid.shape[0] / g_SSDConfig.BATCH_SIZE)

    # optimizer = model_ph_tensor_on_gpu[gpu_id]["optimizer"]
    # reported_loss = model_ph_tensor_on_gpu[gpu_id]["loss"]
    x = model_ph_tensor_on_gpu[gpu_id]["x"]
    y_true_conf = model_ph_tensor_on_gpu[gpu_id]["y_true_conf"]
    y_true_loc = model_ph_tensor_on_gpu[gpu_id]["y_true_loc"]
    conf_loss_mask = model_ph_tensor_on_gpu[gpu_id]["conf_loss_mask"]
    is_training = model_ph_tensor_on_gpu[gpu_id]["is_training"]

    for _ in range(num_batches_valid):
        images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = \
            next(valid_gen)

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
    return valid_loss

def run_training():
    X_train, X_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc = perpare_dataset()

    # Launch the graph
    with tf.Graph().as_default():
        global_step = tf.get_variable("global_step", [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        decay_steps = (len(X_train) // g_SSDConfig.BATCH_SIZE *
                       g_SSDConfig.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(
            learning_rate=g_SSDConfig.INIT_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=g_SSDConfig.LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        opt = g_SSDConfig.OPT(lr)

        # Calculate the gradients for each model descent.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_id in range(g_SSDConfig.GPU_NUMS):
                with tf.device("/gpu:%d" % gpu_id):
                    with tf.name_scope("%s_%d" % (g_SSDConfig.TOWER_NAME,
                                                  gpu_id)) as scope:
                        loss = tower_loss(scope=scope, gpu_id=gpu_id)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            g_SSDConfig.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            if g_SSDConfig.RESUME:
                print('Restoring previously trained model at %s' %
                      g_SSDConfig.MODEL_SAVE_PATH)
                saver.restore(sess, g_SSDConfig.MODEL_SAVE_PATH)

                # Restore previous loss history
                with open(g_SSDConfig.LOSS_HISTORY_PATH, 'rb') as f:
                    loss_history = pickle.load(f)
            else:
                sess.run(tf.global_variables_initializer())
                # [(train_acc_epoch1, valid_acc_epoch1), ...]
                loss_history = []

            # Record time elapsed for performance check
            last_time = time.time()
            train_start_time = time.time()

            # Run NUM_EPOCH epochs of training
            for epoch in range(g_SSDConfig.NUM_EPOCH):
                train_loss_list, valid_loss_list = [], []
                for gpu_id in range(g_SSDConfig.GPU_NUMS):
                    train_loss = run_epoch_on_train_sets(
                        sess, train_op, loss, X_train, y_train_conf,
                        y_train_loc, gpu_id)
                    train_loss_list.append(train_loss)

                    valid_loss = run_epoch_on_valid_sets(
                        sess, loss, X_valid, y_valid_conf, y_valid_loc, gpu_id)
                    valid_loss_list.append(valid_loss)

                # Record and report train/validation/test losses for this epoch
                loss_history.append(
                    (np.mean(train_loss_list), np.mean(valid_loss_list)))

                # Print accuracy every epoch
                print(
                    'Epoch %d -- Train loss: %.4f, Validation loss: %.4f, '
                    'Elapsed time: %.2f sec' % (
                    epoch + 1, loss_history[-1][0], loss_history[-1][1],
                    time.time() - last_time))
                last_time = time.time()

            total_time = time.time() - train_start_time
            print('Total elapsed time: %d min %d sec' % (
                total_time / 60, total_time % 60))

            if g_SSDConfig.SAVE_MODEL:
                # Save model to disk
                save_path = saver.save(sess, g_SSDConfig.MODEL_SAVE_PATH)
                print('Trained model saved at: %s' % save_path)

                # Also save accuracy history
                print('Loss history saved at loss_history.p')
                with open(g_SSDConfig.LOSS_HISTORY_PATH, 'wb') as f:
                    pickle.dump(loss_history, f)

    # Return final test accuracy and accuracy_history
    return loss_history


if __name__ == '__main__':
    run_training()
