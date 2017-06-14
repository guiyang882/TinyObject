#! coding : utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os, json

class DrawLine(object):
    def __init__(self):
        self.filepath = "/Users/liuguiyang/Documents/TrackingDataSet/JL1st/src/0001.jpg"
        self.writepath = self.filepath.replace(".jpg", ".txt")

        self.filepath.split('/')[-1].split('.')
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        self.saveJson = {}
        self.startUp()

    def startUp(self):
        self.points = []
        self.alreadyDrawPos = []
        self.lineInd = 0
        self.img = mpimg.imread(self.filepath)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.imshow(self.img)

    def click(self, event):  # the function to run everytime click the plane
        self.points.append([event.xdata, event.ydata])
        pointsarray = np.asarray(self.points)
        drawMark = self.ax.plot(pointsarray[:, 0], pointsarray[:, 1], 'ro')
        self.alreadyDrawPos.append(drawMark[0])
        self.ax.figure.canvas.draw()

    def on_key_press(self, event):
        '''
        key w: save the line_th
        //key n: start the new line
        //key c: clear the cur line data, which have labled
        key q: close the figure window
        '''
        if event.key in 'w':
            # writer the point into the file
            if len(self.points) <= 0:
                return
            self.saveJson[self.lineInd] = self.points
            self.points = []
            self.lineInd += 1

        if event.key in 'q':
            if len(self.saveJson) == 0:
                return
            with open(self.writepath, 'w') as handle:
                json.dump(self.saveJson, handle)
            plt.close(self.fig)


if __name__ == "__main__":
    draw = DrawLine()
    plt.show()
