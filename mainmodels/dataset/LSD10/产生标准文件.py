# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mainmodels.dataset.LSD10.tools import extract_target_from_xml
from mainmodels.dataset.LSD10.lsd_config import sign_idx_dict, idx_sign_dict


def get_true_target_name(input_name):
    return idx_sign_dict[sign_idx_dict[input_name]]

dir_prefix = "/Volumes/projects/repos/RSI/LSD10/"
anno_dir = dir_prefix + "Annotations/"
save_dir = dir_prefix + "std_results/"

all_test_filepath = dir_prefix + "all_test.txt"
tpl_std_name = "comp4_det_std_{}.txt"


# "target": { "filename": [[pos01], [pos02]]}
all_details = {}
with open(all_test_filepath, "r") as test_reader:
    for line in test_reader:
        img_name = ".".join(line.strip().split(".")[:-1])
        anno_filepath = anno_dir + img_name + ".xml"
        anno_details = extract_target_from_xml(anno_filepath)
        for item in anno_details:
            item[-1] = get_true_target_name(item[-1])
        for item in anno_details:
            target_name = item[-1]
            if target_name not in all_details:
                all_details[target_name] = dict()
            if img_name not in all_details[target_name]:
                all_details[target_name][img_name] = list()
            all_details[target_name][img_name].append(item[:4])
    # 数据组织完成，现在需要分类进行存储
    for key, val in all_details.items():
        with open(save_dir+tpl_std_name.format(key), "w") as f:
            for file_name, target_annos in val.items():
                for annos in target_annos:
                    f.write("{} {} {} {} {} {}\n".format(
                        file_name, 1.0, annos[0], annos[1], annos[2], annos[3]))