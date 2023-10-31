import numpy as np
from graph_solver import *
from g2o_io import *
# from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex
import matplotlib.pyplot as plt


def add(x1, x2):
    # boxplus: self [+] x2
    qnorm = np.linalg.norm(x2[3:])
    if qnorm > 1.0:
        qw, qx, qy, qz = 1., 0., 0., 0.
    else:
        qw = np.sqrt(1. - qnorm**2)
        qx, qy, qz = x2[3:]
    return np.array([x1[0] + x2[0] + 2. * (-(x1[4]**2 + x1[5]**2) * x2[0] + (x1[3] * x1[4] - x1[5] * x1[6]) * x2[1] + (x1[3] * x1[5] + x1[4] * x1[6]) * x2[2]),
                     x1[1] + x2[1] + 2. * ((x1[3] * x1[4] + x1[5] * x1[6]) * x2[0] - (x1[3]**2 + x1[5]**2) * x2[1] + (x1[4] * x1[5] - x1[3] * x1[6]) * x2[2]),
                     x1[2] + x2[2] + 2. * ((x1[3] * x1[5] - x1[4] * x1[6]) * x2[0] + (x1[3] * x1[6] + x1[4] * x1[5]) * x2[1] - (x1[3]**2 + x1[4]**2) * x2[2]),
                     x1[6] * qx + x1[3] * qw + x1[4] * qz - x1[5] * qy,
                     x1[6] * qy - x1[3] * qz + x1[4] * qw + x1[5] * qx,
                     x1[6] * qz + x1[3] * qy - x1[4] * qx + x1[5] * qw,
                     x1[6] * qw - x1[3] * qx - x1[4] * qy - x1[5] * qz])


def sub(x1, x2):
    return np.array([x1[0] - x2[0] + 2. * (-(x2[4]**2 + x2[5]**2) * (x1[0] - x2[0]) + (x2[3] * x2[4] + x2[5] * x2[6]) * (x1[1] - x2[1]) + (x2[3] * x2[5] - x2[4] * x2[6]) * (x1[2] - x2[2])),
                     x1[1] - x2[1] + 2. * ((x2[3] * x2[4] - x2[5] * x2[6]) * (x1[0] - x2[0]) - (x2[3]**2 + x2[5]**2) * (x1[1] - x2[1]) + (x2[4] * x2[5] + x2[3] * x2[6]) * (x1[2] - x2[2])),
                     x1[2] - x2[2] + 2. * ((x2[3] * x2[5] + x2[4] * x2[6]) * (x1[0] - x2[0]) + (x2[4] * x2[5] - x2[3] * x2[6]) * (x1[1] - x2[1]) - (x2[3]**2 + x2[4]**2) * (x1[2] - x2[2])),
                     x2[6] * x1[3] - x2[3] * x1[6] - x2[4] * x1[5] + x2[5] * x1[4],
                     x2[6] * x1[4] + x2[3] * x1[5] - x2[4] * x1[6] - x2[5] * x1[3],
                     x2[6] * x1[5] - x2[3] * x1[4] + x2[4] * x1[3] - x2[5] * x1[6],
                     x2[6] * x1[6] + x2[3] * x1[3] + x2[4] * x1[4] + x2[5] * x1[5]])


def J_x1_minus_x2_dx1(x1, x2):
    return np.array([[1. - 2. * (x2[4]**2 + x2[5]**2), 2. * (x2[3] * x2[4] + x2[5] * x2[6]), 2. * (x2[3] * x2[5] - x2[4] * x2[6]), 0., 0., 0., 0.],
                     [2. * (x2[3] * x2[4] - x2[5] * x2[6]), 1. - 2. * (x2[3]**2 + x2[5]**2), 2. * (x2[4] * x2[5] + x2[3] * x2[6]), 0., 0., 0., 0.],
                     [2. * (x2[3] * x2[5] + x2[4] * x2[6]), 2. * (x2[4] * x2[5] - x2[3] * x2[6]), 1. - 2. * (x2[3]**2 + x2[4]**2), 0., 0., 0., 0.],
                     [0., 0., 0., x2[6], x2[5], -x2[4], -x2[3]],
                     [0., 0., 0., -x2[5], x2[6], x2[3], -x2[4]],
                     [0., 0., 0., x2[4], -x2[3], x2[6], -x2[5]],
                     [0., 0., 0., x2[3], x2[4], x2[5], x2[6]]], dtype=np.float64)


def J_x1_minus_x2_dx2(x1, x2):
    return np.array([[-1. + 2. * (x2[4]**2 + x2[5]**2), -2. * (x2[3] * x2[4] + x2[5] * x2[6]), -2. * (x2[3] * x2[5] - x2[4] * x2[6]), 2. * (x2[4] * (x1[1] - x2[1]) + x2[5] * (x1[2] - x2[2])), 2. * (-2. * x2[4] * (x1[0] - x2[0]) + x2[3] * (x1[1] - x2[1]) - x2[6] * (x1[2] - x2[2])), 2. * (-2. * x2[5] * (x1[0] - x2[0]) + x2[6] * (x1[1] - x2[1]) + x2[3] * (x1[2] - x2[2])), 2. * (x2[5] * (x1[1] - x2[1]) - x2[4] * (x1[2] - x2[2]))],
                     [-2. * (x2[3] * x2[4] - x2[5] * x2[6]), -1. + 2. * (x2[3]**2 + x2[5]**2), -2. * (x2[4] * x2[5] + x2[3] * x2[6]), 2. * (x2[4] * (x1[0] - x2[0]) - 2. * x2[3] * (x1[1] - x2[1]) + x2[6] * (x1[2] - x2[2])), 2. * (x2[3] * (x1[0] - x2[0]) + x2[5] * (x1[2] - x2[2])), 2. * (-x2[6] * (x1[0] - x2[0]) - 2. * x2[5] * (x1[1] - x2[1]) + x2[4] * (x1[2] - x2[2])), 2. * (-x2[5] * (x1[0] - x2[0]) + x2[3] * (x1[2] - x2[2]))],
                     [-2. * (x2[3] * x2[5] + x2[4] * x2[6]), -2. * (x2[4] * x2[5] - x2[3] * x2[6]), -1. + 2. * (x2[3]**2 + x2[4]**2), 2. * (x2[5] * (x1[0] - x2[0]) - x2[6] * (x1[1] - x2[1]) - 2. * x2[3] * (x1[2] - x2[2])), 2. * (x2[6] * (x1[0] - x2[0]) + x2[5] * (x1[1] - x2[1]) - 2. * x2[4] * (x1[2] - x2[2])), 2. * (x2[3] * (x1[0] - x2[0]) + x2[4] * (x1[1] - x2[1])), 2. * (x2[4] * (x1[0] - x2[0]) - x2[3] * (x1[1] - x2[1]))],
                     [0., 0., 0., -x1[6], -x1[5], x1[4], x1[3]],
                     [0., 0., 0., x1[5], -x1[6], -x1[3], x1[4]],
                     [0., 0., 0., -x1[4], x1[3], -x1[6], x1[5]],
                     [0., 0., 0., x1[3], x1[4], x1[5], x1[6]]], dtype=np.float64)


def J_x1_minus_x2_dnewx(x1, x2):
    return    np.array([[-1. + 2. * (x2[4]**2 + x2[5]**2), -2. * (x2[3] * x2[4] + x2[5] * x2[6]), -2. * (x2[3] * x2[5] - x2[4] * x2[6]), 2. * (x2[4] * (x1[1] - x2[1]) + x2[5] * (x1[2] - x2[2])), 2. * (-2. * x2[4] * (x1[0] - x2[0]) + x2[3] * (x1[1] - x2[1]) - x2[6] * (x1[2] - x2[2])), 2. * (-2. * x2[5] * (x1[0] - x2[0]) + x2[6] * (x1[1] - x2[1]) + x2[3] * (x1[2] - x2[2])), 2. * (x2[5] * (x1[1] - x2[1]) - x2[4] * (x1[2] - x2[2]))],
                        [-2. * (x2[3] * x2[4] - x2[5] * x2[6]), -1. + 2. * (x2[3]**2 + x2[5]**2), -2. * (x2[4] * x2[5] + x2[3] * x2[6]), 2. * (x2[4] * (x1[0] - x2[0]) - 2. * x2[3] * (x1[1] - x2[1]) + x2[6] * (x1[2] - x2[2])), 2. * (x2[3] * (x1[0] - x2[0]) + x2[5] * (x1[2] - x2[2])), 2. * (-x2[6] * (x1[0] - x2[0]) - 2. * x2[5] * (x1[1] - x2[1]) + x2[4] * (x1[2] - x2[2])), 2. * (-x2[5] * (x1[0] - x2[0]) + x2[3] * (x1[2] - x2[2]))],
                        [-2. * (x2[3] * x2[5] + x2[4] * x2[6]), -2. * (x2[4] * x2[5] - x2[3] * x2[6]), -1. + 2. * (x2[3]**2 + x2[4]**2), 2. * (x2[5] * (x1[0] - x2[0]) - x2[6] * (x1[1] - x2[1]) - 2. * x2[3] * (x1[2] - x2[2])), 2. * (x2[6] * (x1[0] - x2[0]) + x2[5] * (x1[1] - x2[1]) - 2. * x2[4] * (x1[2] - x2[2])), 2. * (x2[3] * (x1[0] - x2[0]) + x2[4] * (x1[1] - x2[1])), 2. * (x2[4] * (x1[0] - x2[0]) - x2[3] * (x1[1] - x2[1]))],
                        [0., 0., 0., -x1[6], -x1[5], x1[4], x1[3]],
                        [0., 0., 0., x1[5], -x1[6], -x1[3], x1[4]],
                        [0., 0., 0., -x1[4], x1[3], -x1[6], x1[5]]], dtype=np.float64)


def J_plus(x1):
    return np.array([[1. - 2. * (x1[4]**2 + x1[5]**2), 2. * (x1[3] * x1[4] - x1[5] * x1[6]), 2. * (x1[3] * x1[5] + x1[4] * x1[6]), 0., 0., 0.],
                    [2. * (x1[3] * x1[4] + x1[5] * x1[6]), 1. - 2. * (x1[3]**2 + x1[5]**2), 2. * (x1[4] * x1[5] - x1[3] * x1[6]), 0., 0., 0.],
                    [2. * (x1[3] * x1[5] - x1[4] * x1[6]), 2. * (x1[3] * x1[6] + x1[4] * x1[5]), 1. - 2. * (x1[3]**2 + x1[4]**2), 0., 0., 0.],
                    [0., 0., 0., x1[6], -x1[5], x1[4]],
                    [0., 0., 0., x1[5], x1[6], -x1[3]],
                    [0., 0., 0., -x1[4], x1[3], x1[6]],
                    [0., 0., 0., -x1[3], -x1[4], -x1[5]]], dtype=np.float64)


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
        v0 = vertices[self.i].x
        v1 = vertices[self.j].x

        r = sub(self.z, sub(v1, v0))[:6]

        J0 = J_x1_minus_x2_dnewx(self.z, sub(v1, v0)) @ J_x1_minus_x2_dx2(v1, v0) @  J_plus(v0)
        J1 = J_x1_minus_x2_dnewx(self.z, sub(v1, v0)) @ J_x1_minus_x2_dx1(v1, v0) @  J_plus(v1)

        return r, J0, J1


class Pose3Vertex:
    def __init__(self, x):
        self.x = x
        self.size = 6

    def update(self, dx):
        self.x = add(self.x, dx)


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
        #if(vertex[0] == 0):
        #    graph.add_vertex(Pose3Vertex(vertex[1]), is_constant=True)
        #else:
        graph.add_vertex(Pose3Vertex(vertex[1]))  # add vertex to graph
    for edge in edges:
        graph.add_edge(Pose3dbetweenEdge(edge[0][0], edge[0][1], edge[1], edge[2], kernel=kernel))

    draw('before loop-closing', graph)
    graph.solve(step=0)
    draw('after loop-closing', graph)

    plt.show()
