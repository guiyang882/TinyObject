# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/6/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageDraw, ImageFont
import random

def safeInt(ss):
    return int(float(ss))

class BBX:
    def __init__(self):
        pass

    def str2bbx(self, str):
        chrs = str.split(' ')

        self.name = chrs[0]

        self.x = safeInt(chrs[1])
        self.y = safeInt(chrs[2])
        self.w = safeInt(chrs[3])
        self.h = safeInt(chrs[4])
        self.score = float(chrs[5])

    def resize(self, scale, x_d, y_d):
        self.x = safeInt(self.x * scale) + x_d
        self.y = safeInt(self.y * scale) + y_d
        self.w = safeInt(self.w * scale)
        self.h = safeInt(self.h * scale)


class IMGLIB:
    def __init__(self):
        pass

    def setBBXs(self, bboxs, name0):
        self.bbxs = []
        for bbox in bboxs:
            bbx = BBX()
            bbx.name = name0
            bbx.x = safeInt(bbox[0])
            bbx.y = safeInt(bbox[1])
            bbx.w = safeInt(bbox[2])
            bbx.h = safeInt(bbox[3])
            bbx.score = bbox[4]
            self.bbxs.append(bbx)

    def showBBXs(self):
        self.drawBox()
        self.img.show()

    def saveBBXs(self, fileName):
        f = open(fileName, 'w')
        f.write('% bbGt version=3\n')
        for bbx in self.bbxs:
            f.write('%s %d %d %d %d %f 0 0 0 0 0 0\n' % (bbx.name, bbx.x, bbx.y, bbx.w, bbx.h, bbx.score))
        f.close()

    def drawOneBox(self, bbx, thr=-1.0,showName = False):

        if bbx.score >= thr:
            x = bbx.x
            y = bbx.y
            w = bbx.w
            h = bbx.h
            line1 = ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))

            self.draw.line(line1, fill=(255, 0, 0))
            if showName:
                font = ImageFont.truetype("OpenSans-Regular.ttf", 20)
                self.draw.text((x, y - 25), str(bbx.score), fill=(255, 0, 0), font=font)

    def drawBox(self, thr=-1.0, showName = False):
        self.draw = ImageDraw.Draw(self.img)
        for bbx in self.bbxs:
            self.drawOneBox(bbx, thr,showName)

    def read_img(self, fileName):
        self.img = Image.open(fileName)

    def read_ano(self, fileName):

        f = open(fileName, 'r')
        lines = f.readlines()
        self.bbxs = []
        for line in lines[1:]:
            nbbx = BBX()
            nbbx.str2bbx(line)
            self.bbxs.append(nbbx)

            # self.img.show()

    def resizeBBXs(self, r, x_d, y_d):
        for bbx in self.bbxs:
            bbx.resize(r, x_d, y_d)

    def resize(self, width, height, scale=1.0):
        o_width, o_height = self.img.size
        t_width = safeInt(width * scale)
        t_height = safeInt(height * scale)

        o_ratio = o_width / float(o_height)
        n_ratio = width / float(height)

        if o_ratio > n_ratio:
            re_ration = t_width / float(o_width)
            a_height = safeInt(re_ration * o_height)
            a_width = t_width
            self.img = self.img.resize((a_width, a_height), Image.ANTIALIAS)
        else:
            re_ration = t_height / float(o_height)
            a_width = safeInt(re_ration * o_width)
            a_height = t_height
            self.img = self.img.resize((a_width, a_height), Image.ANTIALIAS)

        self.x_d = random.randint(0, abs(a_width - width))
        self.y_d = random.randint(0, abs(a_height - height))
        imgNew = Image.new("RGB", (width, height), "black")

        box = (0, 0, a_width, a_height)
        region = self.img.crop(box)

        imgNew.paste(region, (self.x_d, self.y_d))
        self.img = imgNew
        self.resizeBBXs(re_ration, self.x_d, self.y_d)
        # self.drawBox()

    def cleanAno(self,w0, h0):
        newBBXS = []
        for bbox in self.bbxs:
            if bbox.x >= 0 and bbox.x <= w0 and bbox.y >= 0 and bbox.y <= h0 and bbox.w >= 20 and bbox.w <= w0 and bbox.h >= 30 and bbox.h <= h0:
                bbx = BBX()
                bbx.name = bbox.name
                bbx.x = bbox.x
                bbx.y = bbox.y
                bbx.w = bbox.w
                bbx.h = bbox.h
                bbx.score = bbox.score
                newBBXS.append(bbx)
        self.bbxs = newBBXS

    def save_img(self, imgName):
        self.img.save(imgName)

    def pureResize(self,width, height):
        self.img = self.img.resize((width, height), Image.ANTIALIAS)

if __name__ == '__main__':
    pass

