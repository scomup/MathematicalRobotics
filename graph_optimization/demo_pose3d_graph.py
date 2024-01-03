import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from utilities.plot_tools import *
from utilities import plot_tools
from graph_optimization.graph_solver import *


class Pose3dEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE3 is given in a graph_optimization.md (18)(19)
        """
        Tzx = np.linalg.inv(self.z) @ vertices[self.link[0]].x
        return logSE3(Tzx), [np.eye(6)]


class Pose3dbetweenEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6), kernel=None, color='black'):
        super().__init__(link, z, omega, kernel)
        self.color = color

    def residual(self, vertices):
        """
        The proof of Jocabian of SE3 is given in a graph_optimization.md (18)(19)
        """
        Ti = vertices[self.link[0]].x
        Tj = vertices[self.link[1]].x
        Tij = np.linalg.inv(Ti) @ Tj

        r = logSE3(np.linalg.inv(self.z) @ Tij)

        Tji = np.linalg.inv(Tij)
        Rji, tji = makeRt(Tji)
        J = np.zeros([6, 6])
        J[0:3, 0:3] = -Rji
        J[0:3, 3:6] = -skew(tji) @ Rji
        J[3:6, 3:6] = -Rji
        return r, [J, np.eye(6)]


class Pose3Vertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 6)

    def update(self, dx):
        self.x = self.x @ expSE3(dx)


def draw(figname, graph):
    for n in graph.vertices:
        plot_pose3(figname, n.x, 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    for e in graph.edges:
        if (len(e.link) != 2):
            continue
        i, j = e.link
        _, ti = makeRt((graph.vertices[i].x))
        _, tj = makeRt((graph.vertices[j].x))
        x = [ti[0], tj[0]]
        y = [ti[1], tj[1]]
        z = [ti[2], tj[2]]
        axes.plot(x, y, z, c=e.color, linestyle=':')
    set_axes_equal(figname)


if __name__ == '__main__':
    graph = GraphSolver()

    n = 12
    cur_pose = expSE3(np.array([0, 0, 0, 0, 0, 0]))
    odom = expSE3(np.array([0.2, 0, 0.00, 0.05, 0, 0.5]))
    for i in range(n):
        graph.add_vertex(Pose3Vertex(cur_pose))  # add vertex to graph
        cur_pose = cur_pose @ odom

    graph.add_edge(Pose3dEdge([0], expSE3(np.array([0, 0, 0, 0, 0, 0]))))  # add prior pose to graph

    for i in range(n-1):
        j = (i + 1)
        graph.add_edge(Pose3dbetweenEdge([i, j], odom))  # add edge(i, j) to graph

    graph.add_edge(Pose3dbetweenEdge([n-1, 0], odom, color='red'))

    draw('before loop-closing', graph)
    graph.solve()
    draw('after loop-closing', graph)

    plt.show()
