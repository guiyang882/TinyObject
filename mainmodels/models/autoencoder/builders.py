# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/5/22

from __future__ import absolute_import
from __future__ import print_function

import multiprocessing
import tensorflow as tf
import math

def restore_or_restart(args, paths, sess, global_step):
    pretrained_checkpoint = None
    if args["checkpoint_path"] != '':
        pretrained_checkpoint = tf.train.latest_checkpoint(
            args["checkpoint_path"])
        if not pretrained_checkpoint:
            raise IOError("[E] {} not valid".format(args["checkpoint_path"]))

    if not args["force_restart"]:
        # continue training checkpoint
        continue_checkpoint = tf.train.latest_checkpoint(paths["log"])
        if continue_checkpoint:
            restore_saver = build_restore_saver(
                [global_step], scopes_to_remove=args["exclude_scopes"])
            restore_saver.restore(sess, continue_checkpoint)
        # else if the continue checkpoint does not exists
        # and the pretrained checkpoint has been specified
        # load the weights from the pretrained checkpoint
        elif pretrained_checkpoint:
            restore_saver = build_restore_saver(
                [], scopes_to_remove=args["exclude_scopes"])
            restore_saver.restore(sess, pretrained_checkpoint)
        else:
            print('[!] No checkpoint file found')

def build_restore_saver(variables_to_add=None, scopes_to_remove=None):
    """Return a saver that restores every trainable variable that's not
    under a scope to remove.
    Args:
        variables_to_add: list of variables to add
        scopes_to_remove: list of scopes to remove
    """
    if variables_to_add is None:
        variables_to_add = []

    if scopes_to_remove is None:
        scopes_to_remove = []

    restore_saver = tf.train.Saver(
        variables_to_restore(variables_to_add, scopes_to_remove))
    return restore_saver

def build_train_savers():
    train_saver = tf.train.Saver(max_to_keep=2)
    best_saver = tf.train.Saver(max_to_keep=1)
    return train_saver, best_saver

def build_optimizer(args, global_step):
    initial_lr = float(args['lr'])

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
        initial_lr,
        global_step,
        args["decay_steps"],
        args["decay_rate"],
        staircase=True)
    # Update the learning rate parameter of the optimizer
    args["lr"] = learning_rate

    # Instantiate the optimizer
    optimizer = args["optimizer"](args["lr"])
    return optimizer

def build_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = multiprocessing.cpu_count()
    if num_preprocess_threads > 2:
        num_preprocess_threads -= 2

    if isinstance(label, list):
        row = [image] + label
    else:
        row = [image, label]

    if shuffle:
        return tf.train.shuffle_batch(
            row,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

    return tf.train.batch(
        row,
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

def weight(name,
           shape,
           train_phase,
           initializer=tf.contrib.layers.variance_scaling_initializer(
               factor=2.0,
               mode='FAN_IN',
               uniform=False,
               seed=None,
               dtype=tf.float32),
           wd=0.0):
    weights = tf.get_variable(
        name, shape=shape, initializer=initializer, dtype=tf.float32)

    # show weights of the first layer
    first = len(shape) == 4 and shape[2] in (1, 3, 4)
    if first and train_phase:
        num_kernels = shape[3]
        # check if is a perfect square
        grid_side = math.floor(math.sqrt(num_kernels))
    return weights

def bias(name,
         shape,
         train_phase,
         initializer=tf.constant_initializer(value=0.0)):
    return weight(name, shape, train_phase, initializer=initializer, wd=0.0)

def conv(input_x,
         shape,
         stride,
         padding,
         train_phase,
         bias_term=True,
         activation=tf.identity,
         wd=0.0,
         initializer=tf.contrib.layers.variance_scaling_initializer(
             factor=2.0,
             mode='FAN_IN',
             uniform=False,
             seed=None,
             dtype=tf.float32)):
    """ Define a conv layer.
    Args:
        input_x: a 4D tensor
        shape: weight shape
        stride: a single value supposing equal stride along X and Y
        padding: 'VALID' or 'SAME'
        train_phase: boolean that enables/diables visualizations and train-only specific ops
        bias_term: a boolean to add (if True) the bias term. Usually disable when
                   the layer is wrapped in a batch norm layer
        activation: activation function. Default linear
        train_phase: boolean that enables/diables visualizations and train-only specific ops
        wd: weight decay
        initializer: the initializer to use
    Rerturns:
        op: the conv2d op
    """
    W = weight("W", shape, train_phase, initializer=initializer, wd=wd)
    result = tf.nn.conv2d(input_x, W, [1, stride, stride, 1], padding)
    if bias_term:
        b = bias("b", [shape[3]], train_phase)
        result = tf.nn.bias_add(result, b)

    # apply nonlinearity
    out = activation(result)

    return out

def scale_image(image):
    """Returns the image tensor with values in [-1, 1].
    Args:
        image: [height, width, depth] tensor with values in [0,1]
    """
    image = tf.subtract(image, 0.5)
    # now image has values with zero mean in range [-0.5, 0.5]
    image = tf.multiply(image, 2.0)
    # now image has values with zero mean in range [-1, 1]
    return image