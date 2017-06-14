# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mainmodels.models.ssd.model import SSDModel
from mainmodels.models.ssd.settings import *


FM_ONLY = False  # Only want to see feature map sizes?
with tf.Graph().as_default(), tf.Session() as sess:
    if FM_ONLY:
        if MODEL == "AlexNet":
            from mainmodels.models.ssd.model import AlexNet as MyModel
        else:
            raise NotImplementedError("Model %s not supported" % MODEL)
        _ = MyModel()
    else:
        _ = SSDModel()

    tf.summary.merge_all()
    writer = tf.summary.FileWriter(TENSORBOARD_SAVE_PATH, sess.graph)
    tf.global_variables_initializer().run()
