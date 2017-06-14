# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from mainmodels.models.rpnplus.tools.image_pylib import IMGLIB

if __name__ == '__main__':
    imglib = IMGLIB()

    imageDir = 'image/'

    anoDir = 'ano/'

    saveImgDir = 'saveDir/'
    thr = 0.99
    imageNames = []

    if os.path.isdir(saveImgDir):
        shutil.rmtree(saveImgDir)
    os.mkdir(saveImgDir)

    for file in os.listdir(imageDir):
        file_path = os.path.join(imageDir, file)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[
            1] == '.jpg':
            nameNow = os.path.splitext(file)[0]
            imageName = imageDir + '/' + nameNow + '.jpg'
            anoName = anoDir + '/' + nameNow + '.txt'
            saveImgName = saveImgDir + '/' + nameNow + '.jpg'
            imglib.read_img(imageName)
            imglib.read_ano(anoName)
            imglib.drawBox(thr, False)
            imglib.save_img(saveImgName)
            print(imageName)

