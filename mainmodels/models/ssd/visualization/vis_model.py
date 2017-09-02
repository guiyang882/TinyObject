# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/9/1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mainmodels.models.ssd.ssdmodel import SSDModel


with tf.Graph().as_default(), tf.Session() as sess:
    # This includes the entire graph, e.g. loss function, optimizer, etc.
    _ = SSDModel()

    tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tensorboard_out', sess.graph)
    tf.global_variables_initializer().run()