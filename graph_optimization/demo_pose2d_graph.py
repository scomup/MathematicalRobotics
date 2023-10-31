import numpy as np
from graph_solver import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from utilities.robust_kernel import *


class Pose2dEdge:
    def __init__(self, i, z, omega=None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(3)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (15)(16)
        """
        Tzx = np.linalg.inv(self.z) @ vertices[self.i].x
        return m2v(Tzx), np.eye(3)


class Pose2dbetweenEdge:
    def __init__(self, i, j, z, omega=None, kernel=None, color='black'):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.color = color
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(3)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (15)(16)
        """
        T12 = np.linalg.inv(vertices[self.i].x) @ vertices[self.j].x
        T21 = np.linalg.inv(T12)
        R21, t21 = makeRt(T21)
        J = np.eye(3)
        J[0:2, 0:2] = R21
        J[0:2, 2] = -np.array([-t21[1], t21[0]])
        J = -J
        return m2v(np.linalg.inv(self.z) @ T12), J, np.eye(3)


class Pose2Vertex:
    def __init__(self, x):
        self.x = x
        self.size = 3

    def update(self, dx):
        self.x = self.x @ v2m(dx)


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    for n in graph.vertices:
        plot_pose2(figname, (n.x), 0.05)
    for e in graph.edges:
        if (e.type == 'one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt((graph.vertices[i].x))
        _, tj = makeRt((graph.vertices[j].x))
        x = [ti[0], tj[0]]
        y = [ti[1], tj[1]]
        axes.plot(x, y, c=e.color, linestyle=':')


if __name__ == '__main__':

    graph = GraphSolver()

    n = 12
    cur_pose = v2m(np.array([0, 0, 0]))
    odom = v2m(np.array([0.2, 0, 0.45]))
    for i in range(n):
        graph.add_vertex(Pose2Vertex(cur_pose))  # add vertex to graph
        cur_pose = cur_pose @ odom

    graph.add_edge(Pose2dEdge(0, v2m(np.array([0, 0, 0]))))  # add prior pose to graph

    for i in range(n-1):
        j = (i + 1)
        graph.add_edge(Pose2dbetweenEdge(i, j, odom))  # add edge(i, j) to graph

    graph.add_edge(Pose2dbetweenEdge(n-1, 0, odom, color='red'))

    draw('before loop-closing', graph)
    graph.solve()
    draw('after loop-closing', graph)

    plt.show()
