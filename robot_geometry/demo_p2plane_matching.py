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


class p2planeEdge:
    def __init__(self, i, z, omega=None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if (self.omega is None):
            self.omega = np.eye(1)

    def residual(self, nodes):
        """
        r = P(T(x)*a, plane)
        a: target point
        T: transform matrix, x the se3 of T
        P: point to plane
        """
        x = nodes[self.i].x
        a, plane = self.z
        a_star, dTdx, _ = transform(x, a, True)
        r, dPdT = point2plane(a_star, plane, True)
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

    ref = np.array([[-1, 0, 2.01], [1, 3.02, 1], [-2.1, 3, 1], [1, 0., 1.1], [0, 1, 1.02]])
    tar = np.array([[1.5, 1.5, -1.5], [-1.5, 0.5, -0.5], [2, 2.2, -2]])

    s, plane = find_plane(ref)

    graph.addNode(pose3Node(cur_pose))  # add node to graph
    draw_plane(ax, plane)

    for p in tar:
        graph.addEdge(p2planeEdge(0, [p, plane], kernel=HuberKernel(0.5)))  # add prior pose to graph
        r, j = point2plane(p, plane, True)
        g = -j*r
        draw_arrow(ax, p, g)

    ax.scatter(ref[:, 0], ref[:, 1], ref[:, 2], label='ref')
    ax.scatter(tar[:, 0], tar[:, 1], tar[:, 2], label='tar')
    graph.solve(min_score_change=0.00001, step=0.1)

    x = graph.nodes[0].x

    # for p in tar:
    #     p2 = transform(x, p)
    #     r, j = point2plane(p, plane, True)
    #     g = -j*r
    #     draw_arrow(ax, p2, g)

    R = expSO3(x[0:3])
    t = x[3:6]
    tar2 = (R.dot(tar.T).T + t)
    ax.scatter(tar2[:, 0], tar2[:, 1], tar2[:, 2], label='tar2')

    ax.legend()
    plt.show()
