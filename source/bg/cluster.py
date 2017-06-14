# -*- coding:utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import source.bg.bgutils as bgutils

class ImageCluster():
    isKMeans, isMiniBathKMeans = False, True

    filepath = "/Volumes/LargeDisk/TrackingDataSet/Iris_Vancouver/part01/0001.jpg"
    image = cv2.imread(filepath)
    width, height = image.shape[0], image.shape[1]

    linefile = filepath.replace(".jpg", ".txt")

    # image = cv2.imread("/Users/liuguiyang/Documents/TrackingDataSet/Iris_Vancouver/part02/0001.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image[0:width, 0:int(width * 1.5)]
    height, width = image.shape[0], image.shape[1]
    print("width: ", width, " height: ", height)
    # Reshape the image to be a list of pixels
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))

    def getLine(self):
        lines = []
        with open(self.linefile, 'r') as handle:
            jsondata = json.load(handle)
            for key in jsondata.keys():
                lines.append(jsondata[key])
        self.linepoints = lines

    def getRect(self, curX, curY):
        local_width, local_height = 50, 50

        leftx, lefty = curX - local_width, curY - local_height
        if leftx < 0: leftx = 0
        if lefty < 0: lefty = 0
        leftPos = (int(leftx), int(lefty))

        rightx, righty = curX + local_width, curY + local_height
        if rightx > self.width: rightx = self.width
        if righty > self.height: righty = self.height
        rightPos = (int(rightx), int(righty))

        localSubImg = self.image[leftPos[1]:rightPos[1], leftPos[0]:rightPos[0]]

        return (int(leftx), int(lefty)), (int(rightx), int(righty)), localSubImg

    def getSubLineRect(self, curX, curY, nextX, nextY, isMarkLine = False):
        width_delta = 35
        leftx = min(curX, nextX) - width_delta
        lefty = min(curY, nextY)
        if leftx < 0: leftx = 0
        if lefty < 0: lefty = 0
        leftPos = (int(leftx), int(lefty))

        rightx = max(curX, nextX) + width_delta
        righty = max(curY, nextY)
        if rightx > self.width: rightx = self.width
        if righty > self.height: righty = self.height
        rightPos = (int(rightx), int(righty))

        localSubImg = self.image[leftPos[1]:rightPos[1], leftPos[0]:rightPos[0]]

        ## mark the point in the localSubImage
        p0x, p0y = curX - leftPos[0], curY - leftPos[1]
        p1x, p1y = nextX - leftPos[0], nextY - leftPos[1]
        if isMarkLine:
            cv2.line(localSubImg, (int(p0x), int(p0y)), (int(p1x), int(p1y)), thickness=2, color=(255,0,0))
        return (int(leftx), int(lefty)), (int(rightx), int(righty)), localSubImg, (p0x, p0y), (p1x, p1y)

    def lcoal_cluster(self, localImg):
        localImg_array = localImg.reshape((localImg.shape[0] * localImg.shape[1], 3))
        if self.isKMeans:
            # Clusters the pixels
            clt = KMeans(n_clusters=3)
            y_pred = clt.fit(localImg_array)

        if self.isMiniBathKMeans:
            clt = MiniBatchKMeans(n_clusters=3)
            y_pred = clt.fit(localImg_array)

        if self.isKMeans or self.isMiniBathKMeans:
            lables_pred = y_pred.labels_.reshape((localImg.shape[0], localImg.shape[1]))
            return lables_pred, clt.cluster_centers_
        return None, None

    def local_segment(self):
        self.getLine()

        for line in self.linepoints:
            for i in range(len(line)-1):
                plt.clf()

                curX, curY = line[i][0], line[i][1]
                nextX, nextY = line[i+1][0], line[i+1][1]

                print(int(curX), int(curY))
                print(int(nextX), int(nextY))

                theta = bgutils.getAngle((curX, curY), (nextX, nextY))
                ## 返回方框在原始图像中的位置信息leftPos, rightPos
                ## 返回取出的局部图像的信息localSubImg
                ## 返回标记的道路在局部图像中的相对位置信息nPos1, nPos2
                if 0:
                    savePrefix = "../../output/part01/"
                    leftPos, rightPos, localSubImg, nPos1, nPos2 = self.getSubLineRect(curX, curY, nextX, nextY,
                                                                                       isMarkLine=False)
                    cv2.imwrite(savePrefix + "no_local" + str(i) + ".jpg", localSubImg,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    leftPos, rightPos, localSubImg, nPos1, nPos2 = self.getSubLineRect(curX, curY, nextX, nextY,
                                                                                       isMarkLine=True)
                    cv2.imwrite(savePrefix + "local" + str(i) + ".jpg", localSubImg,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    with open(savePrefix + "line" + str(i) + ".txt", 'w') as handle:
                        print >> handle, nPos1[0], nPos1[1]
                        print >> handle, nPos2[0], nPos2[1]
                    continue
                else:
                    leftPos, rightPos, localSubImg, nPos1, nPos2 = self.getSubLineRect(curX, curY, nextX, nextY,
                                                                                       isMarkLine=False)

                tmpImg = self.image.copy()
                cv2.rectangle(tmpImg, leftPos, rightPos, (255, 0, 0), 2)
                cv2.circle(tmpImg, leftPos, 1, (0, 0, 255), 2)
                cv2.circle(tmpImg, rightPos, 1, (0, 0, 255), 2)
                cv2.circle(tmpImg, (int(curX), int(curY)), 1, (0, 0, 255), 2)
                cv2.circle(tmpImg, (int(nextX), int(nextY)), 1, (0, 255, 0), 2)

                ## Run The Cluster Info
                if localSubImg.shape[1] != 0 and localSubImg.shape[0] != 0:
                    lables_pred, lables_centers = self.lcoal_cluster(localSubImg)
                    single_lable_list = bgutils.split_lables(lables_pred)
                    k = len(single_lable_list)

                    plt.subplot(2, (k+3) // 2, 1)
                    plt.imshow(localSubImg)
                    plt.title("Local")
                    for i in range(k):
                        plt.subplot(2, (k+3)//2, i+2)
                        plt.imshow(single_lable_list[i])
                        plt.title("cluster " + str(i))
                    plt.subplot(2, (k+3)//2, k+2)
                    plt.imshow(lables_pred)
                    plt.title("total")
                    plt.show(block=True)

                    ## Run find the counters info
                    # hierarchy : [Next, Previous, First_Child, Parent]
                    subImg = localSubImg.copy()
                    if 1:
                        tmp_subImg = np.zeros((localSubImg.shape[0], localSubImg.shape[1]), dtype=np.uint8)
                        for i in range(len(single_lable_list)):
                            tmp = single_lable_list[i]
                            im2, contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(tmp_subImg, contours, -1, 1, 1)

                        im2, contours, hierarchy = cv2.findContours(tmp_subImg, cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(subImg, contours, -1, (255, 0, 0), 1)
                    else:
                        lables_pred = lables_pred.astype(np.uint8)
                        im2, contours, hierarchy = cv2.findContours(lables_pred, cv2.RETR_TREE,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(subImg, contours, -1, (255, 0, 0), 1)
                    cv2.line(subImg, bgutils.getIntPoint(nPos1), bgutils.getIntPoint(nPos2), color=(0, 255, 0), thickness=1)
                    plt.imshow(subImg)
                    plt.title("Local Cluster With Line")
                    plt.show()

                    if 0:
                        ## show the cluster block into the source image
                        lable_block = bgutils.convert2Dto3D(lables_pred, lables_centers)
                        tmpImg[leftPos[1]:rightPos[1], leftPos[0]:rightPos[0]] = lable_block
                        plt.imshow(tmpImg)
                        plt.title("Source local Cluster")
                        plt.show()

                    if 0:
                        ## extractor the hog feature
                        fd, hog_image = bgutils.hogFeature(localSubImg)
                        fig, (ax1, ax2) = plt.subplots(1, 2)

                        ax1.axis('off')
                        ax1.imshow(localSubImg)
                        ax1.set_title('Input image')
                        ax1.set_adjustable('box-forced')

                        ax2.axis('off')
                        ax2.imshow(hog_image, cmap=plt.cm.gray)
                        ax2.set_title('Histogram of Oriented Gradients')
                        ax1.set_adjustable('box-forced')
                        plt.show()

                    # return

    def global_detectRoad(self):
        if self.isKMeans:
            # Clusters the pixels
            clt = KMeans(n_clusters=5)
            y_pred = clt.fit(self.image_array)

        if self.isMiniBathKMeans:
            clt = MiniBatchKMeans(n_clusters=10)
            y_pred = clt.fit(self.image_array)

        if self.isKMeans or self.isMiniBathKMeans:
            t_lables = np.unique(y_pred.labels_)
            print(t_lables)
            lables_pred = y_pred.labels_.reshape((self.image.shape[0], self.image.shape[1]))
            plt.figure(1)
            plt.imshow(lables_pred)
            plt.show()

            for k in t_lables:
                a = lables_pred.copy()
                for i in range(lables_pred.shape[0]):
                    for j in range(lables_pred.shape[1]):
                        if k == lables_pred[i][j]:
                            a[i][j] = 1
                        else:
                            a[i][j] = 0
                plt.imshow(a)
                plt.imsave("test0" + str(k) + ".jpg", a)
                plt.show()

if __name__ == '__main__':
    obj = ImageCluster()
    obj.local_segment()