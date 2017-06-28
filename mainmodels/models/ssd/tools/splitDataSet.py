# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
这个文件主要是用来重新划分数据集，将221类目标，缩小到10类目标，这样可以训练的更快一些
'''
import os
from mainmodels.models.ssd.settings import g_SSDConfig

