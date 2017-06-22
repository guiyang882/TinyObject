# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

proj_root = "/".join(os.path.abspath(__file__).split("/")[0:-5])
main_models = "/".join([proj_root, "mainmodels"])
models = "/".join([proj_root, "models"])
print(proj_root)

sys.path.extend([proj_root, main_models, models])
print(sys.path)