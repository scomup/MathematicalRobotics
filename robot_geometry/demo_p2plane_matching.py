import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities import *
from guass_newton_method.guass_newton import *
from geometry_plot import *
from basic_geometry import *
from graph_optimization.graph_solver import *
from slam.reprojection import *
from utilities.robust_kernel import *
from demo_p2line_matching import transform, plus


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


def residual(T, param):
    """
    r = P(T(x)*a, plane)
    a: target point
    T: transform matrix, x the se3 of T
    P: point to plane
    """
    a, plane = param
    a_star, dTdx, _ = transform(T, a, True)
    r, dPdT = point2plane(a_star, plane, True)
    J = dPdT.dot(dTdx)
    return r*np.ones(1), J.reshape([1, 6])


def plus(T, delta):
    return T @ p2m(delta)


if __name__ == '__main__':

    fig = plt.figure("plane", figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')

    graph = GraphSolver()
    T = p2m(np.array([0, 0, 0, 0, 0, 0]))

    ref = np.array([[-1, 0, 2.01], [1, 3.02, 1], [-2.1, 3, 1], [1, 0., 1.1], [0, 1, 1.02]])
    src = np.array([[1.5, 1.5, -1.5], [-1.5, 0.5, -0.5], [2, 2.2, -2]])

    s, plane = find_plane(ref)

    draw_plane(ax, plane)

    for p in src:
        r, j = point2plane(p, plane, True)
        g = -j*r
        draw_arrow(ax, p, g)

    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], label='target points (plane)')
    ax.scatter(src[:, 0], src[:, 1], src[:, 2], label='source points')

    params = []
    for i in src:
        params.append([i, plane])

    gn = guassNewton(6, residual, params, plus, kernel=HuberKernel(0.5))

    T = gn.solve(T, step=0.1)

    R, t = makeRt(T)
    tar2 = (R.dot(src.T).T + t)
    ax.scatter(tar2[:, 0], tar2[:, 1], tar2[:, 2], label='matched source points')

    ax.legend()
    plt.show()
