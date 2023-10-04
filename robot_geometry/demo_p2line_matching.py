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


class p2lineEdge:
    def __init__(self, i, z, omega=None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(1)

    def residual(self, nodes):
        """
        r = P(T(x)*a, c, dir)
        a: target point
        T: transform matrix, x the se3 of T
        c: center of line
        dir: direction of line
        P: point to line
        """
        x = nodes[self.i].x
        a, c, dir = self.z
        a_star, dTdx, _ = transform(x, a, True)
        r, dPdT = point2line(a_star, c, dir, True)
        # dPdT2 = numericalDerivative(point2line, [a_star, c, dir], 0)
        J = dPdT.dot(dTdx)
        return r*np.ones(1), J.reshape([1, 6])


class pose3Node:
    def __init__(self, x):
        self.x = x
        self.size = x.size

    def update(self, dx):
        self.x = pose_plus(self.x, dx)


if __name__ == '__main__':

    fig = plt.figure("plane", figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')

    graph = graphSolver()
    cur_pose = np.array([0, 0, 0, 0, 0, 0])

    # ref = np.array([[0.1, 0.2, -0.2], [1, 1.02, 0.1], [2.1, 2.5, 0.4], [2.8, 3.0, 0.5], [4.2, 3.9, 1.2]])
    # tar = np.array([[1.5, 1.5, 1.2], [0.4, 0.7, 0.5], [2, 2.2, 2]])
    ref = np.array([[0.1, 0.2, -0.0], [1, 1.02, 0.0], [2.1, 2.5, 0.0], [2.8, 3.0, 0.0], [4.2, 3.9, 0]])
    # tar = np.array([[1.5, 1.5, 1.2], [0.4, 0.7, 0.5], [2, 2.2, 2]])
    tar = np.array([[1.6, 1.5, 1.5], [0.5, 0.4, 0.5], [2, 2.2, 1.9]])

    s, center, direction = find_line(ref)
    # tar = np.array([[0.1, 0.2, -0], [1, 1.02, 0], [2.1, 2.5, 0], [2.8, 3.0, 0], [4.2, 3.9, 0], [5, 5, 0], [5, 5, 0]])

    graph.addNode(pose3Node(cur_pose))  # add node to graph
    draw_line(ax, center, direction)

    for p in tar:
        graph.addEdge(p2lineEdge(0, [p, center, direction], kernel=HuberKernel(0.5)))  # add prior pose to graph
        r, j = point2line(p, center, direction, True)
        g = -j*r
        draw_arrow(ax, p, g)

    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], label='ref')
    ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], label='tar')
    graph.solve(min_score_change=0.00001, step=0.1)
    x = graph.nodes[0].x

    # R = expSO3(x[0:3])
    # t = x[3:6]
    # tar2 = (R.dot(tar.T) + t).T
    for p in tar:
        p2 = transform(x, p)
        r, j = point2line(p2, center, direction, True)
        g = -j*r
        draw_arrow(ax, p2, g)

    R = expSO3(x[0:3])
    t = x[3:6]
    tar2 = (R.dot(tar.T).T + t)
    ax.scatter(tar2[:, 0], tar2[:, 1], tar2[:, 2], label='tar2')
    ax.legend()
    plt.show()
