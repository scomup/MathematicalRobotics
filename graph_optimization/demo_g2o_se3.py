import numpy as np
from graph_solver import *
from g2o_io import *
# from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex
import matplotlib.pyplot as plt


class Pose3dbetweenEdge:
    def __init__(self, i, j, z, omega=None,  kernel=None, color='black'):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.color = color
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(self.z.shape[0])

    def residual(self, vertices):
        T1 = expSE3(vertices[self.i].x)
        T2 = expSE3(vertices[self.j].x)

        T12 = np.linalg.inv(T1).dot(T2)
        T21 = np.linalg.inv(T12)
        R, t = makeRt(T21)
        J = np.zeros([6, 6])
        J[0:3, 0:3] = R
        J[0:3, 3:6] = skew(t).dot(R)
        J[3:6, 3:6] = R
        J = -J
        return logSE3(np.linalg.inv(expSE3(self.z)).dot(T12)), J, np.eye(6)


class Pose3Vertex:
    def __init__(self, x):
        self.x = x
        self.size = 6

    def update(self, dx):
        self.x = logSE3(expSE3(self.x) @ expSE3(dx))


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.add_subplot(projection='3d')
    vertices = []
    edges = []
    for v in graph.vertices:
        vertices.append(v.x[:3])
    for e in graph.edges:
        edges.append([*(graph.vertices[e.i].x)[:3]])
        edges.append([*(graph.vertices[e.j].x)[:3]])
    vertices = np.array(vertices)
    axes.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],  s=10, color='k')
    edges = np.array(edges)
    axes.plot(xs=edges[:, 0], ys=edges[:, 1], zs=edges[:, 2], c='b', linewidth=1)


"""
def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.add_subplot(projection='3d')
    vertices = []
    edges = []
    for v in graph.vertices:
        vertices.append(expSE3(v.x)[0:3, 3])
    for e in graph.edges:
        edges.append([*expSE3(graph.vertices[e.i].x)[0:3, 3]])
        edges.append([*expSE3(graph.vertices[e.j].x)[0:3, 3]])
    vertices = np.array(vertices)
    axes.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],  s=10, color='k')
    edges = np.array(edges)
    axes.plot(xs=edges[:, 0], ys=edges[:, 1], zs=edges[:, 2], c='b', linewidth=1)
"""


if __name__ == '__main__':

    graph = GraphSolver(use_sparse=True)

    edges, vertices = load_g2o_se3('data/g2o/sphere2500.g2o')
    kernel = None
    # kernel = HuberKernel(1)
    for vertex in vertices:
        if(vertex[0] == 0):
            graph.add_vertex(Pose3Vertex(vertex[1]), is_constant=True)
        else:
            graph.add_vertex(Pose3Vertex(vertex[1]))  # add vertex to graph
    for edge in edges:
        graph.add_edge(Pose3dbetweenEdge(edge[0][0], edge[0][1], edge[1], edge[2], kernel=kernel))

    # draw('before loop-closing', graph)
    graph.solve(step=0)
    # draw('after loop-closing', graph)

    plt.show()
