import numpy as np
import matplotlib.pyplot as plt
from mathR.optimization.gauss_newton import *
from mathR.utilities.robust_kernel import *
from mathR.utilities.math_tools import *
from mathR.utilities.pcd_io import load_pcd
import sys
import os


class Plot3D:
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


def residual(T, param):
    """
    The residual vector of 3d point matching is given by gauss_newton_method.md (7)
    The proof of Jocabian of 3d point matching is given in a gauss_newton_method.md (12)
    """
    a, b = param
    R, t = makeRt(T)
    r = R.dot(a) + t - b
    I = np.eye(3)
    j = np.zeros([4, 6])
    j[0:3, 0:3] = I
    j[0:3, 3:6] = skew(-a)
    j = (T @ j)[0:3]
    return r.flatten(), j


def plus(T, delta):
    """
    The incremental function of SE3 is given in gauss_newton_method.md (5)
    """
    return T @ expSE3(delta)


if __name__ == '__main__':

    plt3d = Plot3D()

    T_truth = expSE3(np.array([1000, -0.1, 0.1, 2.1, 2.2, -1.3]))
    pcd_path = os.path.join(os.path.dirname(__file__), "../data/bunny.pcd")
    a = load_pcd(pcd_path)
    # a = (np.random.rand(elements, 3)-0.5)*2
    b = transform3d(T_truth, a.T).T
    elements = a.shape[0]
    b += np.random.normal(0, 0.001, (elements, 3))

    params = []
    for i in range(a.shape[0]):
        params.append([a[i], b[i]])

    gn = GaussNewton(6, residual, params, plus)
    T_cur = expSE3(np.array([0., 0., 0., 0., 0., 0.]))
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
