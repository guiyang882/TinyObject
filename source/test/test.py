# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/7/21

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import math

K = input()
need = []
for i in range(K):
    t = input()
    need = map(int, t.strip().split(","))
matrix = []
for i in range(K):
    tmp = input()
    matrix.append(map(int, tmp.strip().split(",")))

res = 0
lastX, lastY = K-1, -1
for i in range(K):
    for j in range(len(matrix[i])):
        if sum(need) > 0 and matrix[i][j] and need[i]:
            res += (math.fabs(i-lastX) + math.fabs(j-lastY))
            need[i] -= 1
            lastX, lastY = i, j
print(res)