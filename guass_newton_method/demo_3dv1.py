import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from guass_newton import *
from demo_3d import Plot3D


def residual(T, param):
    a, b = param
    R, t = makeRt(T)
    r = R.dot(a) + t - b
    j = np.array([[1, 0, 0, 0, a[2], -a[1]],
                  [0, 1, 0, -a[2], 0, a[0]],
                  [0, 0, 1, a[1], -a[0], 0]])
    j = R.dot(j)
    return r.flatten(), j


def plus(T, delta):
    return T @ p2m(delta)


if __name__ == '__main__':
    plt3d = Plot3D()
    T_truth = p2m(np.array([1000, -0.1, 0.1, 2.1, 2.2, -1.3]))
    elements = 100
    a = (np.random.rand(elements, 3)-0.5)*2
    b = transform3d(T_truth, a.T).T
    b += np.random.normal(0, 0.03, (elements, 3))

    params = []
    for i in range(a.shape[0]):
        params.append([a[i], b[i]])

    gn = guassNewton(6, residual, params, plus)
    T_cur = p2m(np.array([0., 0., 0., 0., 0., 0.]))
    cur_a = a.copy()
    last_score = None
    itr = 0
    while(True):
        plt3d.update(cur_a, b)
        dx, score = gn.solve_once(T_cur)
        T_cur = gn.plus(T_cur, dx)
        cur_a = transform3d(T_cur, a.T).T
        print('iter %d: %f' % (itr, score))
        itr += 1
        if (last_score is None):
            last_score = score
            continue
        if (last_score < score):
            break
        if (last_score - score < 0.0001):
            break
        last_score = score
    plt3d.show()
