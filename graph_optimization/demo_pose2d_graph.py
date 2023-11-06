import numpy as np
from graph_solver import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from utilities.robust_kernel import *


class Pose2dEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(3), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (13)(14)
        """
        Tzx = np.linalg.inv(self.z) @ vertices[self.link[0]].x
        return m2v(Tzx), [np.eye(3)]


class Pose2dbetweenEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(3), kernel=None, color='black'):
        super().__init__(link, z, omega, kernel)
        self.color = color

    def residual(self, vertices):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (13)(14)
        """
        T0 = vertices[self.link[0]].x
        T1 = vertices[self.link[1]].x
        T01 = np.linalg.inv(T0) @ T1

        r = m2v(np.linalg.inv(self.z) @ T01)

        T10 = np.linalg.inv(T01)
        R10, t10 = makeRt(T10)
        J = np.eye(3)
        J[0:2, 0:2] = R10
        J[0:2, 2] = -np.array([-t10[1], t10[0]])
        J = -J
        return r, [J, np.eye(3)]


class Pose2Vertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 3)

    def update(self, dx):
        self.x = self.x @ v2m(dx)


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    for n in graph.vertices:
        plot_pose2(figname, (n.x), 0.05)
    for e in graph.edges:
        if (len(e.link) != 2):
            continue
        i, j = e.link
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

    graph.add_edge(Pose2dEdge([0], v2m(np.array([0, 0, 0]))))  # add prior pose to graph

    for i in range(n-1):
        j = (i + 1)
        graph.add_edge(Pose2dbetweenEdge([i, j], odom))  # add edge(i, j) to graph

    graph.add_edge(Pose2dbetweenEdge([n-1, 0], odom, color='red'))

    draw('before loop-closing', graph)
    graph.solve()
    draw('after loop-closing', graph)

    plt.show()