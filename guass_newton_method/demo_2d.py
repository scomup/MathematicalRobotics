import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import *
from guass_newton import *


class Plot2D:
    def __init__(self):
        # self.fig = plt.figure()
        self.fig, self.ax = plt.subplots()

    def update(self, a, b):
        plt.cla()
        self.ax.scatter(a[:, 0], a[:, 1], c='r')
        self.ax.scatter(b[:, 0], b[:, 1], c='b')
        plt.pause(0.2)

    def show(self):
        plt.show()


def residual(T, param):
    """
    The residual vector of 2d point matching is given by guass_newton_method.md (7)
    The proof of Jocabian of 2d point matching is given in a guass_newton_method.md (12)
    """
    a, b = param
    R, t = makeRt(T)
    r = R.dot(a) + t - b
    j = np.array([[1, 0, -a[1]],
                  [0, 1,  a[0]]])
    j = R.dot(j)
    return r.flatten(), j


def plus(T, delta):
    """
    The incremental function of SE2 is given in guass_newton_method.md (5)
    """
    return T @ v2m(delta)

if __name__ == '__main__':

    plt2d = Plot2D()
    T_truth = v2m(np.array([-0.3, 0.2, np.pi/2]))
    elements = 10
    # a = (np.random.rand(elements, 2)-0.5)*2

    a = np.array([[0, 0], [0, 0.5], [0, 1], [0.5, 0], [0.5, 1], [1, 0], [1, 0.5], [1, 1]])
    b = transform2d(T_truth, a.T).T
    b[0, 1] -= 0.5
    b[0, 0] += 0.5

    params = []
    for i in range(a.shape[0]):
        params.append([a[i], b[i]])

    # b += np.random.normal(0, 0.03, (elements, 2))
    kernel = CauchyKernel(0.1)
    gn = guassNewton(3, residual, params, plus, kernel)
    T_cur = v2m(np.array([0., 0., 0.]))
    cur_a = a.copy()
    last_score = None
    itr = 0
    while(True):
        plt2d.update(cur_a, b)
        dx, score = gn.solve_once(T_cur)
        T_cur = gn.plus(T_cur, dx)
        cur_a = transform2d(T_cur, a.T).T
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
    plt2d.show()
