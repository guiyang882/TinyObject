# -*- coding: utf-8 -*-

import numpy as np
import importlib

import os
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages")
importlib.reload(sys)

import cv2
import matplotlib.pylab as plt

class VideoTracking():
    def __init__(self):
        self.video_dir = "/Volumes/projects/第三方数据下载/长光卫星高清视频第二批/处理的视频"
        self.video_name = \
            "20170602-美国-明尼阿波利斯01.mp4"
        self.cap, self.prev, self.cur = None, None, None
        self.isFirst = True
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture("/".join([self.video_dir, self.video_name]))
        self.writer = None
        fps = int(self.cam.get(cv2.CAP_PROP_FPS))
        size = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(fps, size)
        self.frameInd = 1

    def splitImages(self, frame, blockNums = 4):
        width, height = frame.shape[0], frame.shape[1]
        c_x, c_y = width // 2, height // 2

        subImgs = []
        subImgs.append([0, c_x, 0, c_y, frame[0:c_x, 0:c_y]])
        subImgs.append([c_x, width, 0, c_y, frame[c_x:width, 0:c_y]])
        subImgs.append([0, c_x, c_y, height, frame[0:c_x, c_y:height]])
        subImgs.append([c_x, width, c_y, height, frame[c_x:width, c_y:height]])
        return subImgs

    def run(self) -> object:
        # 获得视频的格式
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if ret == False:
                break
            subImages = self.splitImages(frame, 4)
            if False:
                for i in range(4):
                    plt.subplot(2, 2, i+1)
                    plt.imshow(subImages[i])
                plt.show()
            if True:
                save_prefix = "/".join([self.video_dir,
                                        "SRC_"+self.video_name.split(".")[0]])
                if not os.path.isdir(save_prefix):
                    os.makedirs(save_prefix)

                for i in range(4):
                    filename = "%d_%d_%d_%d_%04d.png" % (subImages[i][0],
                                                         subImages[i][1],
                                                         subImages[i][2],
                                                         subImages[i][3],
                                                         self.frameInd)
                    filepath = "/".join([save_prefix, filename])
                    cv2.imwrite(filepath, subImages[i][4])
                    print(filepath)
                self.frameInd += 1
        self.cam.release()

if __name__ == "__main__":
    obj = VideoTracking()
    obj.run()
