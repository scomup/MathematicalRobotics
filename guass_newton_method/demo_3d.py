import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from guass_newton import *


class plot3D:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        # elev = -107.
        # azim = 0
        # self.ax.view_init(elev, azim)

    def update(self, a, b):
        plt.cla()
        self.ax.scatter3D(a[:, 0], a[:, 1], a[:, 2], c='r')
        self.ax.scatter3D(b[:, 0], b[:, 1], b[:, 2], c='b')
        plt.pause(0.2)

    def show(self):
        plt.show()


def residual(x, param):
    """
    The residual vector of 3d point matching is given by guass_newton_method.md (7)
    The proof of Jocabian of 3d point matching is given in a guass_newton_method.md (12)
    """
    a, b = param
    T0 = expSE3(x)
    R, t = makeRt(T0)
    r = R.dot(a) + t - b
    I = np.eye(3)
    j = np.zeros([4, 6])
    j[0:3, 0:3] = I
    j[0:3, 3:6] = skew(-a)
    j = (T0 @ j)[0:3]
    return r.flatten(), j


def plus(x1, x2):
    """
    The incremental function of SE3 is given in guass_newton_method.md (5)
    """
    return logSE3(expSE3(x1).dot(expSE3(x2)))


if __name__ == '__main__':

    plt3d = plot3D()

    x_truth = np.array([1000, -0.1, 0.1, 2.1, 2.2, -1.3])
    a = np.loadtxt("data/bunny.txt")
    # a = (np.random.rand(elements, 3)-0.5)*2
    b = transform3d(x_truth, a.T).T
    elements = a.shape[0]
    b += np.random.normal(0, 0.001, (elements, 3))

    params = []
    for i in range(a.shape[0]):
        params.append([a[i], b[i]])

    gn = guassNewton(residual, params, plus)
    x_cur = np.array([0., 0., 0., 0., 0., 0.])
    cur_a = a.copy()
    last_score = None
    iter = 0
    while(True):
        plt3d.update(cur_a, b)
        dx, score = gn.solve_once(x_cur)
        x_cur = gn.plus(x_cur, dx)
        cur_a = transform3d(x_cur, a.T).T
        print('iter %d: %f' % (iter, score))
        iter += 1
        if(last_score is None):
            last_score = score
            continue
        if(last_score < score):
            break
        if(last_score - score < 0.0001):
            break
        last_score = score
    plt3d.show()
