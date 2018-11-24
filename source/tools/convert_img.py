# encoding: utf-8
"""
@contact: liuguiyang15@mails.ucas.edu.cn
@file: convert_img.py.py
@time: 2018/5/23 23:22
"""

import cv2
from matplotlib import pyplot as plt


img_path = "/Users/liuguiyang/Desktop/000011_612_204_1124_716_7.jpg"
img_name = "000011_612_204_1124_716_7.jpg"

image = cv2.imread(img_path)

xImg = cv2.flip(image, 1, dst=None)
xImg1 = cv2.flip(image, 0, dst=None)
xImg2 = cv2.flip(image, -1, dst=None)

cv2.imwrite("{}_{}".format("flip1", img_name), xImg)
cv2.imwrite("{}_{}".format("flip0", img_name), xImg1)
cv2.imwrite("{}_{}".format("flip-1", img_name), xImg2)


h, w = image.shape[:2]
crop_h, crop_w = h - 50, w - 50
import random

for i in range(3):
    x = random.randint(0, 50)
    y = random.randint(0, 50)
    crop_image = image[y:y+crop_h, x:x+crop_w]
    crop_image = cv2.resize(crop_image, (w, h))
    cv2.imwrite("{}_{}_{}".format(x, y, img_name), crop_image)



# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

# 将原图旋转不同角度
rotated = rotate(image, 45)
cv2.imshow("Rotated by 45 Degrees", rotated)
cv2.imwrite("{}_{}".format("r45", img_name), rotated)
rotated = rotate(image, -45)
cv2.imshow("Rotated by -45 Degrees", rotated)
cv2.imwrite("{}_{}".format("r-45", img_name), rotated)
rotated = rotate(image, 90)
cv2.imshow("Rotated by 90 Degrees", rotated)
cv2.imwrite("{}_{}".format("r90", img_name), rotated)
rotated = rotate(image, -90)
cv2.imshow("Rotated by -90 Degrees", rotated)
cv2.imwrite("{}_{}".format("r-90", img_name), rotated)
cv2.waitKey(0)