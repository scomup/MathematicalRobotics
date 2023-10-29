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
                p = expSE2(arr)
                v = [int(nums[0]), p]
                vertices.append(v)
                continue
            if line.startswith("EDGE_SE2"):
                nums = line[9:].split()
                arr = np.array([float(m) for m in nums[2:]], dtype=np.float64)
                link = [int(nums[0]), int(nums[1])]
                estimate = expSE2(arr)
                information = upper_matrix_to_full(arr[3:], 3)
                e = [link, information, estimate]
                edges.append(e)
                continue
    return edges, vertices


if __name__ == '__main__':
    load_g2o_se2('data/g2o/manhattanOlson3500.g2o')


