# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/23

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

proj_dir = "/".join(os.path.abspath(__file__).split("/")[:-4])
print(proj_dir)
mainmodels_path = "/".join([proj_dir, "mainmodels"])
sys.path.append(proj_dir)
sys.path.append(mainmodels_path)