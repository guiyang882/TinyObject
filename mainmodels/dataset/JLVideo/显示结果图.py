# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

from mainmodels.dataset import tools
from mainmodels.dataset.LSD10.tools import extract_target_from_xml

dir_prefix = "/Volumes/projects/repos/RSI/CSUVOCFormat/608x608/large_000013363_total/"
anno_dir = dir_prefix + "Annotations/"
image_dir = dir_prefix + "JPEGImages/"
model_res_dir = dir_prefix + "results/"
all_test_filepath = dir_prefix + "all_test.txt"

all_dir_prefix = "/Volumes/projects/repos/RSI/CSUVOCFormat/source/large_000013363_total/"
all_anno_dir = all_dir_prefix + "Annotations/"
all_image_dir = all_dir_prefix + "JPEGImages/"

res_save_dir = "/Volumes/projects/repos/RSI/CSUVOCFormat/608x608" \
               "/large_000013363_total/images2video_res/"

def merge_model_results(prob=0.50):
    # {"f_name": [[prob,area,target],]}
    all_model_results = dict()
    for filename in os.listdir(model_res_dir):
        target_name = filename.split(".")[0].split("_")[-1]
        model_res_path = model_res_dir + filename
        with open(model_res_path, "r") as res_reader:
            for line in res_reader:
                line = line.strip().split(" ")
                if float(line[1]) < prob:
                    continue
                f_name, t_prob = line[0]+".jpg", float(line[1])
                t_area = [int(float(a)) for a in line[2:]]
                if f_name not in all_model_results:
                    all_model_results[f_name] = list()
                all_model_results[f_name].append([t_prob]+t_area+[target_name])
    return all_model_results

all_model_results = merge_model_results()

# cv2.namedWindow("000013363", cv2.WINDOW_NORMAL)
# for idx in range(1, len(os.listdir(all_anno_dir)), 5):
#     anno_name = "%06d.xml" % idx
#     src_img_name = "%06d.jpg" % idx
#     anno_path = all_anno_dir + anno_name
#     img_path = all_image_dir + src_img_name
#     src_image = cv2.imread(img_path)
#     targets_info = tools.extract_airplane_posinfo(anno_path)
#     if len(targets_info) == 0:
#         continue
#
#     src_img = cv2.imread(img_path)
#     for area in targets_info:
#         cv2.rectangle(src_img,
#                       (area[0], area[1]),
#                       (area[2], area[3]),
#                       (0, 255, 0), 2)
#
#     for sub_img_name in all_model_results.keys():
#         if sub_img_name[:6] == src_img_name[:6]:
#             x0, y0, x1, y1 = [int(a) for a in sub_img_name[7:-4].split("_")][:4]
#             for items in all_model_results[sub_img_name]:
#                 cv2.rectangle(src_img,
#                               (items[1]+x0, items[2]+y0),
#                               (items[3]+x0, items[4]+y0),
#                               (0, 0, 255), 2)
#     cv2.imwrite(res_save_dir+src_img_name, src_img,
#                 [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imshow("000013363", src_img)
#     cv2.waitKey()


# with open(all_test_filepath, "r") as test_reader:
#     for line in test_reader:
#         filename = image_dir + line.strip()
#         gt_anno_path = anno_dir + ".".join(
#             line.strip().split(".")[:-1] + ["xml"])
#         anno_details = extract_target_from_xml(gt_anno_path)
#         # print(anno_details)
#         src_img = cv2.imread(filename)
#         for area in anno_details:
#             cv2.rectangle(src_img,
#                           (area[0], area[1]),
#                           (area[2], area[3]),
#                           (0, 255, 0), 2)
#         # 绘制模型检测结果的目标位置
#         # print(all_model_results[line.strip()])
#         for items in all_model_results[line.strip()]:
#             cv2.rectangle(src_img,
#                           (items[1], items[2]),
#                           (items[3], items[4]),
#                           (0, 0, 255), 2)
#         cv2.imshow(line.strip(), src_img)
#         cv2.waitKey()

img_root = '/Volumes/projects/repos/RSI/CSUVOCFormat/608x608' \
           '/large_000013363_total/images2video_res/'
fps = 5    #保存视频的FPS，可以适当调整

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi',
                              fourcc, fps, (4096, 3072))#最后一个是保存图片的尺寸

for i in range(1, 673, 5):
    img_path = img_root + "%06d.jpg" % i
    if not os.path.exists(img_path):
        continue
    print(img_path)
    frame = cv2.imread(img_path)
    videoWriter.write(frame)
videoWriter.release()