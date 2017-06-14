import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype="uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
					  color.astype("uint8").tolist(), -1)
		startX = endX
		print(color)
	# return the bar chart
	return bar

def getAngle(curPos, nextPos):
    c_x, c_y = int(curPos[0]), int(curPos[1])
    n_x, n_y = int(nextPos[0]), int(nextPos[1])
    if n_y - n_x == 0:
        return np.pi / 2
    return np.arctan(1.0 * (n_y - c_y) / 1.0 * (n_x - c_x))

def split_lables(lables_pred):
	c_lables = np.unique(lables_pred)
	res = []
	for k in c_lables:
		tmp = lables_pred.copy()
		tmp = tmp.astype(np.uint8)
		for i in range(lables_pred.shape[0]):
			for j in range(lables_pred.shape[1]):
				if lables_pred[i][j] == k:
					tmp[i][j] = 1
				else:
					tmp[i][j] = 0
		res.append(tmp)
	return res

def convert2Dto3D(lables_array, lables_centers):
	h, w = lables_array.shape
	bar = np.zeros((h, w, 3), dtype="uint8")
	colors = np.unique(lables_array)
	for i in range(h):
		for j in range(w):
			ind = lables_array[i][j]
			bar[i][j] = lables_centers[ind].astype(np.uint8)
	return bar

def hogFeature(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return fd, hog_image

def getIntPoint(fPos):
	return (int(fPos[0]), int(fPos[1]))