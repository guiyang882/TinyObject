# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/11/10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 5-tennis court,
# 6-basketball court, 7-ground track field, 8-harbor, 9-bridge, 10-vehicle
sign_idx_dict = {
    "car": 1,
    "truck": 2,
    "tractor": 4,
    "campingcar": 5,
    "van": 9,
    "vehicle": 10,
    "PickUp": 11,
    "ship": 23,
    "airplane": 31,
}

remove_label_ids = [7, 8]

idx_sign_dict = {
    1: "car",
    2: "truck",
    4: "tractor",
    5: "campingcar",
    9: "van",
    10: "vehicle",
    11: "pickup",
    23: "ship",
    31: "airplane"
}
