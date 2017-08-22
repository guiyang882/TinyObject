# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/8/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, codecs, json
import cv2

dir_prefix = "/Volumes/projects/第三方数据下载/JL1ST" \
             "/SRC_JL101B_MSS_20160904180811_000013363_101_001_L1B_MSS_SSD/"
for img_file_name in os.listdir(dir_prefix + "src/"):
    if img_file_name.startswith("._"):
        continue
    print(img_file_name)
    anno_file = ".".join(img_file_name.split(".")[:-1] + ["json"])
    print(anno_file)
    src_img = cv2.imread(dir_prefix+"src/"+img_file_name)
    with codecs.open(dir_prefix+"annotation/"+anno_file, "r") as handle:
        anno_targets = json.load(handle)
        for area in anno_targets:
            print(area)
            cv2.rectangle(src_img, (area[0], area[1]), (area[2], area[3]),
                          (0, 255, 0), 2)
    cv2.imshow("dsfdsf", src_img)
    cv2.waitKey()

