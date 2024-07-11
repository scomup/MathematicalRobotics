# Copyright (c) 2020 Jeff Irion and contributors
#
# This file originated from the `graphslam` package:
#
#   https://github.com/JeffLIrion/python-graphslam

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def upper_matrix_to_full(arr, n):
    mat = np.zeros((n, n), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            mat[i, j] = arr[idx]
            mat[j, i] = arr[idx]
            idx += 1
    return mat


def load_g2o_se2(infile):
    edges = []
    vertices = []
    with open(infile) as f:
        for line in f.readlines():
            if line.startswith("VERTEX_SE2"):
                nums = line[10:].split()
                arr = np.array([float(n) for n in nums[1:]], dtype=np.float64)
                T = v2m(arr)
                v = [int(nums[0]), T]
                vertices.append(v)
                continue
            if line.startswith("EDGE_SE2"):
                nums = line[9:].split()
                arr = np.array([float(m) for m in nums[2:]], dtype=np.float64)
                link = [int(nums[0]), int(nums[1])]
                estimate = v2m(arr[:3])
                information = upper_matrix_to_full(arr[3:], 3)
                e = [link, estimate, information]
                edges.append(e)
                continue
    return edges, vertices


def load_g2o_pose_quat(infile):
    edges = []
    vertices = []
    with open(infile) as f:
        for line in f.readlines():
            if line.startswith("VERTEX_SE3:QUAT"):
                nums = line[16:].split()
                arr = np.array([float(n) for n in nums[1:]], dtype=np.float64)
                v = [int(nums[0]), arr[:7]]
                vertices.append(v)
                continue

            if line.startswith("EDGE_SE3:QUAT"):
                nums = line[14:].split()
                arr = np.array([float(n) for n in nums[2:]], dtype=np.float64)
                link = [int(nums[0]), int(nums[1])]
                information = upper_matrix_to_full(arr[7:], 6)
                e = [link, arr[:7], information]
                edges.append(e)
                continue
    return edges, vertices


def load_g2o_se3(infile):
    edges = []
    vertices = []
    with open(infile) as f:
        for line in f.readlines():
            if line.startswith("VERTEX_SE3:QUAT"):
                nums = line[16:].split()
                arr = np.array([float(n) for n in nums[1:]], dtype=np.float64)
                R = quaternion_to_matrix(arr[3:7])
                t = arr[:3]
                T = makeT(R, t)
                v = [int(nums[0]), T]
                vertices.append(v)
                continue

            if line.startswith("EDGE_SE3:QUAT"):
                nums = line[14:].split()
                arr = np.array([float(n) for n in nums[2:]], dtype=np.float64)
                link = [int(nums[0]), int(nums[1])]
                R = quaternion_to_matrix(arr[3:7])
                t = arr[:3]
                T = makeT(R, t)
                information = upper_matrix_to_full(arr[7:], 6)
                e = [link, T, information]
                edges.append(e)
                continue
    return edges, vertices

if __name__ == '__main__':
    load_g2o_se2('data/g2o/manhattanOlson3500.g2o')
    load_g2o_se3('data/g2o/sphere2500.g2o')

