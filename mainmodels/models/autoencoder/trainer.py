# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/5/22

from __future__ import absolute_import
from __future__ import print_function

import time

import tensorflow as tf
from models.autoencoder.data_input import WordData
from trainmodel.models.autoencoder.builders import build_optimizer
from trainmodel.models.autoencoder.builders import build_train_savers
from trainmodel.models.autoencoder.model import SingleLayerAE

from mainmodels.models.autoencoder.utils import InputType


class WordAETrainer():

    def __init__(self, model):
        self._model = model

    def train(self, dataset, args):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')

            with tf.device("/cpu:0"):
                print(type(InputType.train))
                images, labels = dataset.inputs(
                    input_type=InputType.train,
                    batch_size=args["batch_size"]
                )
            with tf.device("/cpu:0"):
                is_training_, reconstructions = self._model.get(
                    images=images,
                    train_phase=True,
                    l2_penalty=args["l2"]
                )

                loss = self._model.loss(reconstructions, labels)

                optimizer = build_optimizer(args=args,
                                            global_step=global_step)
                train_op = optimizer.minimize(
                    loss,
                    global_step=global_step,
                )

                init = [
                    tf.variables_initializer(tf.global_variables() + tf.local_variables()),
                    tf.tables_initializer()
                ]

                with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    sess.run(init)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    # Create the Savers
                    train_saver, best_saver = build_train_savers()
                    old_gs = sess.run([global_step])

                    # Restart from where we were
                    for step in range(old_gs, args["epochs"]+1):
                        start_time = time.time()
                        _, loss_value = sess.run(
                            [train_op, loss], feed_dict={is_training_: True}
                        )
                        duration = time.time() - start_time
                        print(duration, loss_value)

                    coord.request_stop()
                    coord.join(threads)

if __name__ == '__main__':
    args = {
        "epochs": 10,
        "batch_size": 32,
        "l2": 1e-5,
        "lr": 1E-3,
        "decay_steps": 100000,
        "decay_rate": 0.96,
        "optimizer": tf.train.AdamOptimizer,
        "paths": {
            "basedir": "/Users/liuguiyang/Documents/CodeProj/PyProj/TinyObjectDetection/trainmodel/log/autoencoder/",
            "best": "best/",
            "train": "train/",
            "valid": "valid/"
        }
    }
    slae = SingleLayerAE()
    obj = WordAETrainer(slae)
    dataset = WordData()
    obj.train(dataset=dataset, args=args)