# -*- coding: utf-8 -*-

import numpy as np
import cv2

class VideoTracking():
    def __init__(self):
        #video_dir = "/Volumes/projects/TrainData/Stanford_Drone_Dataset/stanford_campus_dataset/videos/bookstore/video0/"
        self.video_dir = "/Users/liuguiyang/Documents/TrackingDataSet/Iris/"
        self.cap, self.prev, self.cur = None, None, None
        self.isFirst = True
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(self.video_dir + "Iris_Dubai_Crop.avi")
        self.frame_idx = 0
        self.lk_params = dict(winSize=(7, 7),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=5,
                      blockSize=5)

    def run(self):
        # 获得视频的格式
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if ret == False:
                break
            width, height = frame.shape[0], frame.shape[1]
            if self.isFirst:
                self.prev = cv2.cvtColor(frame[0:width, 0:height], cv2.COLOR_RGB2GRAY)
                self.isFirst = False
                continue

            cur_rgb = frame[0:width, 0:height]
            self.cur = cv2.cvtColor(cur_rgb, cv2.COLOR_RGB2GRAY)

            if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                img0, img1 = self.prev, self.cur
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                       **self.lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                        **self.lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
                good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(cur_rgb, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(cur_rgb, [np.int32(tr) for tr in self.tracks], False,
                              (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线

            if self.frame_idx % self.detect_interval == 0:  # 每5帧检测一次特征点
                mask = np.zeros_like(self.cur)  # 初始化和视频大小相同的图像
                mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  # 跟踪的角点画圆
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(self.cur, mask=mask, **self.feature_params)  # 像素级别角点检测
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

            self.frame_idx += 1
            self.prev = self.cur
            cv2.imshow('lk_track', cur_rgb)
            ch = cv2.waitKey()
            if ch == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = VideoTracking()
    obj.run()
