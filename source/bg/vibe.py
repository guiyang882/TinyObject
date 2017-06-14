import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

COLOR_BACKGROUND = 0
COLOR_FOREGROUND = 255

class VIBEBaseModel:
    def __init__(self):
        self.width = None
        self.height = None
        self.numberOfSamples = 10
        self.matchingThreshold = 20
        self.matchingNumber = 2
        self.updateFactor = 16

        self.historyImage = []
        self.historyBuffer = []
        self.NUMBER_OF_HISTORY_IMAGES = 2
        self.lastHistoryImageSwapped = 0

        self.jump = []
        self.neighbor = []
        self.position = []

    def __str__(self):
        print("Using ViBe background subtraction algorithm:")
        print("- Number of samples per pixel:", self.numberOfSamples)
        print("- Number of matches needed:", self.matchingNumber)
        print("- Matching threshold:", self.matchingThreshold)
        print("- Model update subsampling factor:", self.updateFactor)

    def setImageSize(self, width, height):
        self.width, self.height = width, height

    def setNumSamples(self, numOfSamples):
        self.numberOfSamples = numOfSamples

    def setUpdateFactor(self, updateFactor):
        self.updateFactor = updateFactor

    def setMatchNumber(self, matchNum):
        self.matchingNumber = matchNum

    def setMatchThreshold(self, matchThreshold):
        self.matchingThreshold = matchThreshold


class VIBE(VIBEBaseModel):

    def sequential_AllocInit_8UC1R(self, imageData, width, height):
        def plus_noise():
            tmpFrame = imageData.copy().astype(np.int8)
            tmpMask = np.random.randint(0, 20, tmpFrame.shape[:2], dtype=np.int8) - 10
            tmpFrame += tmpMask
            tmpFrame[tmpFrame < 0] = 0
            tmpFrame[tmpFrame > 255] = 255
            return tmpFrame.astype(np.uint8)

        self.setImageSize(width, height)
        if len(self.historyImage) > 0:
            self.historyImage = []
        for i in range(self.NUMBER_OF_HISTORY_IMAGES):
            self.historyImage.append(imageData.copy())
        for i in range(self.numberOfSamples - self.NUMBER_OF_HISTORY_IMAGES):
            self.historyBuffer.append(plus_noise())
        # fills the buffers with random values
        fillsize = max(self.width, self.height) * 2 + 1
        self.jump = np.random.randint(0, 2*self.updateFactor, fillsize, dtype=np.uint32) + 1
        self.neighbor = np.random.randint(0, 3, fillsize, dtype=np.int32) - 1 + (np.random.randint(0, 3, fillsize, dtype=np.int32) - 1) * self.width
        self.position = np.random.randint(0, self.numberOfSamples, fillsize, dtype=np.uint32)

    def sequential_Segmentation_8UC1R(self, imageData):
        def getRowCol(indexPos):
            nr = indexPos // self.width
            nc = indexPos % self.width
            return nr, nc

        def getNthRowCol(indexPos):
            nth = indexPos // (self.width * self.height)
            indexPos -= nth * self.width * self.height
            nr = indexPos // self.width
            nc = indexPos % self.width
            return nth, nr, nc

        segmentation_map = imageData.copy()
        segmentation_map.setflags(write=True)
        segmentation_map[:] = self.matchingNumber - 1
        # for first history image
        tmp = abs(imageData - self.historyImage[0])
        segmentation_map[tmp > self.matchingThreshold] = self.matchingNumber
        # for next history image
        for i in range(1, self.NUMBER_OF_HISTORY_IMAGES):
            tmp = abs(imageData - self.historyImage[i])
            segmentation_map[tmp <= self.matchingThreshold] -= 1
        # for swapping
        self.lastHistoryImageSwapped = (self.lastHistoryImageSwapped + 1) % self.NUMBER_OF_HISTORY_IMAGES
        swappingImageBuffer = self.historyImage[self.lastHistoryImageSwapped]

        # now we move the buffer and leave thee historyImages
        numberOfTests = self.numberOfSamples - self.NUMBER_OF_HISTORY_IMAGES - 1
        for index in range(self.width*self.height-1, -1, -1):
            r, c = getRowCol(index)
            if segmentation_map[r, c] <= 0:
                continue

            indexHistoryBuffer = index * numberOfTests
            currentValue = imageData[r, c]

            for i in range(numberOfTests, 0, -1):
                nth, nr, nc = getNthRowCol(indexHistoryBuffer)
                if abs(int(currentValue) - int(self.historyBuffer[nth][nr, nc])) <= self.matchingThreshold:
                    segmentation_map[r, c] -= 1

                    # swaping: Putting found value in history image buffer
                    tmp = swappingImageBuffer[r, c]
                    swappingImageBuffer[r, c] = self.historyBuffer[nth][nr, nc]
                    self.historyBuffer[nth][nr, nc] = tmp

                    # exit the inner loop
                    if segmentation_map[r, c] <= 0:
                        break
                indexHistoryBuffer += 1
        segmentation_map[segmentation_map > 0] = COLOR_FOREGROUND
        return segmentation_map

    def sequential_Update_8UC1R(self, imageData, updating_mask):
        def getNthRowCol(indexPos):
            nth = indexPos // (self.width * self.height)
            indexPos = indexPos - nth * (self.width * self.height)
            nr = indexPos // self.width
            nc = indexPos % self.width
            return nth, nr, nc

        numberOfTests = self.numberOfSamples - self.NUMBER_OF_HISTORY_IMAGES
        # updating All the frame. except the border
        for y in range(1, self.height-1):
            shift = np.random.randint(0, self.width)
            indX = self.jump[shift]
            while indX < self.width - 1:
                if updating_mask[y, indX] == COLOR_BACKGROUND:
                    val = imageData[y, indX]
                    index_neighbor = indX + y * self.width + self.neighbor[shift]
                    nth, nr, nc = getNthRowCol(index_neighbor)
                    if self.position[shift] < self.NUMBER_OF_HISTORY_IMAGES:
                        self.historyImage[self.position[shift]][y, indX] = val
                        self.historyImage[nth][nr, nc] = val
                    else:
                        pos = self.position[shift] - self.NUMBER_OF_HISTORY_IMAGES
                        t_index = indX + y * self.width
                        t_index = t_index * numberOfTests + pos
                        nth, nr, nc = getNthRowCol(t_index)
                        self.historyBuffer[nth][nr, nc] = val
                        nth, nr, nc = getNthRowCol(index_neighbor * numberOfTests+ pos)
                        self.historyBuffer[nth][nr, nc] = val
                shift += 1
                indX += self.jump[shift]
        # for first row
        y, shift = 0, np.random.randint(0, self.width)
        indX = self.jump[shift]
        while indX <= self.width - 1:
            index = indX + y * self.width
            if updating_mask[y, indX] == COLOR_BACKGROUND:
                if self.position[shift] < self.NUMBER_OF_HISTORY_IMAGES:
                    self.historyImage[self.position[shift]][y, indX] = imageData[y, indX]
                else:
                    pos = self.position[shift] - self.NUMBER_OF_HISTORY_IMAGES
                    nth, nr, nc = getNthRowCol(index * numberOfTests + pos)
                    self.historyBuffer[nth][nr, nc] = imageData[nr, nc]
            shift += 1
            indX += self.jump[shift]

        # for last row
        y, shift = self.height - 1, np.random.randint(0, self.width)
        indX = self.jump[shift]
        while indX <= self.width - 1:
            index = indX + y * self.width
            if updating_mask[y, indX] == COLOR_BACKGROUND:
                if self.position[shift] < self.NUMBER_OF_HISTORY_IMAGES:
                    self.historyImage[self.position[shift]][y, indX] = imageData[y, indX]
                else:
                    pos = self.position[shift] - self.NUMBER_OF_HISTORY_IMAGES
                    nth, nr, nc = getNthRowCol(index * numberOfTests + pos)
                    self.historyBuffer[nth][nr, nc] = imageData[y, indX]
            shift += 1
            indX += self.jump[shift]

        # for first column
        x, shift = 0, np.random.randint(0, self.height)
        indY = self.jump[shift]
        while indY <= self.height - 1:
            index = x + indY * self.width
            if updating_mask[indY, x] == COLOR_BACKGROUND:
                if self.position[shift] < self.NUMBER_OF_HISTORY_IMAGES:
                    self.historyImage[self.position[shift]][indY, x] = imageData[indY, x]
                else:
                    pos = self.position[shift] - self.NUMBER_OF_HISTORY_IMAGES
                    nth, nr, nc = getNthRowCol(index * numberOfTests + pos)
                    self.historyBuffer[nth][nr, nc] = imageData[nr, nc]
            shift += 1
            indY += self.jump[shift]

        # for last column
        x, shift = self.width - 1, np.random.randint(0, self.height)
        indY = self.jump[shift]
        while indY <= self.height - 1:
            index = x + indY * self.width
            if updating_mask[indY, x] == COLOR_BACKGROUND:
                if self.position[shift] < self.NUMBER_OF_HISTORY_IMAGES:
                    self.historyImage[self.position[shift]][indY, x] = imageData[indY, x]
                else:
                    pos = self.position[shift] - self.NUMBER_OF_HISTORY_IMAGES
                    nth, nr, nc = getNthRowCol(index * numberOfTests + pos)
                    self.historyBuffer[nth][nr, nc] = imageData[indY, x]
            shift += 1
            indY += self.jump[shift]

        # the first pixel
        if np.random.randint(0, self.updateFactor) == 0:
            if updating_mask[0, 0] == 0:
                pos = np.random.randint(0, self.numberOfSamples)
                if pos < self.NUMBER_OF_HISTORY_IMAGES:
                    self.historyImage[pos][0, 0] = imageData[0, 0]
                else:
                    pos -= self.NUMBER_OF_HISTORY_IMAGES
                    self.historyBuffer[pos][0, 0] = imageData[0, 0]

if __name__ == '__main__':
    obj = VIBE()
    isFirst = True
    dirpath = "/Volumes/Ubuntu/TrackingDataSet/JL1st/src/part01/"
    imagelist = os.listdir(dirpath)
    segmentationMap = None
    # cv2.namedWindow("Source Image", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Segmentation Vibe", cv2.WINDOW_NORMAL)
    plt.figure(1)
    ax1, ax2 = plt.subplot(121), plt.subplot(122)
    plt.ion()

    for item in imagelist:
        if item.endswith(".jpg") == False:
            continue
        imagepath = dirpath + item
        print(imagepath)
        imageData = cv2.imread(imagepath)
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
        imageData = imageData[200:imageData.shape[0]-200, 200:imageData.shape[1]-200]
        height, width = imageData.shape[:2]
        if isFirst == True:
            obj.sequential_AllocInit_8UC1R(imageData, width, height)
            segmentationMap = imageData.copy()
            isFirst = False

        segmentationMap = obj.sequential_Segmentation_8UC1R(imageData)
        obj.sequential_Update_8UC1R(imageData, segmentationMap)

        '''
        Post-processes the segmentation map. This step is not compulsory.
        Note that we strongly recommend to use post-processing filters, as they
        always smooth the segmentation map. For example, the post-processing fileter
        used for the Change Detection dataset is a 5*5 median filter
        '''
        segmentationMap = cv2.medianBlur(segmentationMap, 3)
        if 1:
            plt.sca(ax1)
            plt.imshow(imageData, cmap=plt.cm.gray)
            plt.title("Source Image")
            plt.sca(ax2)
            plt.imshow(segmentationMap,  cmap=plt.cm.gray)
            plt.title("Segmentation Vibe")
            plt.pause(0.05)
        if 0:
            cv2.imshow("Source Image", imageData)
            cv2.imshow("Segmentation Vibe", segmentationMap)
            cv2.waitKey(1)
