# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""本文件就是将训练完毕的模型提供稳定的服务"""

import numpy as np
import tensorflow as tf

from models.tradition.cnn import model as CNNModel
from models.tradition.multicnn import model as MultiCNNModel
from models.tradition.config import g_CNNConfig
from models.tradition.config import g_MultiCNNConfig


class ModelServer(object):

    def __init__(self, use_model_flag=0):

        self._use_model_flag = use_model_flag
        self._images = None
        self._logits = None
        self._sess = None
        if use_model_flag == 0:
            self._train_log_dir = g_CNNConfig.train_log_dir
            self._image_height = g_CNNConfig.image_height
            self._image_width = g_CNNConfig.image_width
            self._image_depth = g_CNNConfig.image_depth
        elif use_model_flag == 1:
            self._train_log_dir = g_MultiCNNConfig.train_log_dir
            self._image_height = g_MultiCNNConfig.image_height
            self._image_width = g_MultiCNNConfig.image_width
            self._image_depth = g_MultiCNNConfig.image_depth
        else:
            raise IOError("There is not this model in the system !")

        if not tf.gfile.Exists(self._train_log_dir):
            raise IOError(self._train_log_dir + " model ckpt file not found !")

    def build_model(self):
        self._images = tf.placeholder(dtype=tf.float32,
                                      shape=(None, self._image_height,
                                             self._image_width,
                                             self._image_depth))

        if self._use_model_flag == 0:
            softmax_linear = CNNModel.inference(self._images)
        elif self._use_model_flag == 1:
            softmax_linear = MultiCNNModel.inference(self._images)
        else:
            raise IOError("There is not this model in the system !")
        return softmax_linear

    def load_model(self):
        with tf.Graph().as_default():
            self._logits = self.build_model()
            saver = tf.train.Saver()

            self._sess = tf.Session()
            ckpt = tf.train.get_checkpoint_state(self._train_log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)
                # for node in tf.get_default_graph().as_graph_def().node:
                #     if "Assign" in node.name or "restore_all" in node.name or \
                #                     "report" in node.name or "save" in node.name or \
                #                     "GradientDescent" in node.name or "gradients" in \
                #             node.name:
                #         continue
                #     print(node.name)
                graph = tf.get_default_graph()
                self._logits = graph.get_tensor_by_name(
                    "softmax_linear/softmax_linear:0")
                print(self._logits)
            else:
                raise IOError("No checkpoint file found !")

    def predict(self, images):
        if self._sess is None:
            raise IOError("TF Model not loaded !")
        # 输入的images是一个[Batch, height, width, channel] Numpy for [0-255]
        # np.float32
        images = np.divide(images, 255.0)
        images = np.subtract(images, 0.5)
        images = np.multiply(images, 2)
        pred = self._sess.run(self._logits,
                              feed_dict={self._images: images})
        return pred
