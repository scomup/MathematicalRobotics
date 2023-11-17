import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import *
from guass_newton_method.guass_newton import *
from geometry_plot import *
from basic_geometry import *
from utilities.robust_kernel import *


def transform(T, p, calcJ=False):
    R, t = makeRt(T)
    r = R.dot(p) + t
    if (calcJ is True):
        M = R.dot(skew(-p))
        dTdx = np.hstack([R, M])
        dTdp = R
        return r, dTdx, dTdp
    else:
        return r


def plus(T, delta):
    return T @ p2m(delta)


def residual(T, param):
    """
    r = P(T(x)*a, c, dir)
    a: target point
    T: transform matrix, x the se3 of T
    c: center of line
    dir: direction of line
    P: point to line
    """
    a, c, dir = param
    a_star, dTdx, _ = transform(T, a, True)
    r, dPdT = point2line(a_star, c, dir, True)
    J = dPdT @ dTdx
    return r * np.ones(1), J.reshape([1, 6])


if __name__ == '__main__':

    fig = plt.figure("plane", figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')

    T = p2m(np.array([0, 0, 0, 0, 0, 0]))

    tar = np.array([[0.1, 0.2, -0.0], [1, 1.02, 0.0], [2.1, 2.5, 0.0], [2.8, 3.0, 0.0], [4.2, 3.9, 0]])
    src = np.array([[1.6, 1.5, 1.5], [0.5, 0.4, 0.5], [2, 2.2, 1.9]])

    _, center, direction = find_line(tar)

    params = []
    for i in src:
        params.append([i, center, direction])

    gn = guassNewton(6, residual, params, plus, kernel=HuberKernel(2))

    T = gn.solve(T, step=0.1)

    draw_line(ax, center, direction)

    for p in src:
        r, j = point2line(p, center, direction, True)
        g = -j*r
        draw_arrow(ax, p, g)

    ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], label='target points (line)')
    ax.scatter(src[:, 0], src[:, 1], src[:, 2], label='source points')

    for p in src:
        p2 = transform(T, p)
        r, j = point2line(p2, center, direction, True)
        g = -j*r
        draw_arrow(ax, p2, g)

    R, t = makeRt(T)
    tar2 = (R.dot(src.T).T + t)
    ax.scatter(tar2[:, 0], tar2[:, 1], tar2[:, 2], label='matched source points')
    ax.legend()
    plt.show()
