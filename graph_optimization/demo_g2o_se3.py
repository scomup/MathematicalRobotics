import numpy as np
from graph_solver import *
from g2o_io import *
# from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex
import matplotlib.pyplot as plt

def add(self, other):
    # boxplus: self [+] other
    qnorm = np.linalg.norm(other[3:])
    if qnorm > 1.0:
        qw, qx, qy, qz = 1., 0., 0., 0.
    else:
        qw = np.sqrt(1. - qnorm**2)
        qx, qy, qz = other[3:]
    return np.array([self[0] + other[0] + 2. * (-(self[4]**2 + self[5]**2) * other[0] + (self[3] * self[4] - self[5] * self[6]) * other[1] + (self[3] * self[5] + self[4] * self[6]) * other[2]),
                    self[1] + other[1] + 2. * ((self[3] * self[4] + self[5] * self[6]) * other[0] - (self[3]**2 + self[5]**2) * other[1] + (self[4] * self[5] - self[3] * self[6]) * other[2]),
                    self[2] + other[2] + 2. * ((self[3] * self[5] - self[4] * self[6]) * other[0] + (self[3] * self[6] + self[4] * self[5]) * other[1] - (self[3]**2 + self[4]**2) * other[2]),
                    self[6] * qx + self[3] * qw + self[4] * qz - self[5] * qy,
                    self[6] * qy - self[3] * qz + self[4] * qw + self[5] * qx,
                    self[6] * qz + self[3] * qy - self[4] * qx + self[5] * qw,
                    self[6] * qw - self[3] * qx - self[4] * qy - self[5] * qz])

def sub(self, other):
    return np.array([self[0] - other[0] + 2. * (-(other[4]**2 + other[5]**2) * (self[0] - other[0]) + (other[3] * other[4] + other[5] * other[6]) * (self[1] - other[1]) + (other[3] * other[5] - other[4] * other[6]) * (self[2] - other[2])),
                    self[1] - other[1] + 2. * ((other[3] * other[4] - other[5] * other[6]) * (self[0] - other[0]) - (other[3]**2 + other[5]**2) * (self[1] - other[1]) + (other[4] * other[5] + other[3] * other[6]) * (self[2] - other[2])),
                    self[2] - other[2] + 2. * ((other[3] * other[5] + other[4] * other[6]) * (self[0] - other[0]) + (other[4] * other[5] - other[3] * other[6]) * (self[1] - other[1]) - (other[3]**2 + other[4]**2) * (self[2] - other[2])),
                    other[6] * self[3] - other[3] * self[6] - other[4] * self[5] + other[5] * self[4],
                    other[6] * self[4] + other[3] * self[5] - other[4] * self[6] - other[5] * self[3],
                    other[6] * self[5] - other[3] * self[4] + other[4] * self[3] - other[5] * self[6],
                    other[6] * self[6] + other[3] * self[3] + other[4] * self[4] + other[5] * self[5]])


def jacobian_self_ominus_other_wrt_self(self, other):
    return np.array([[1. - 2. * (other[4]**2 + other[5]**2), 2. * (other[3] * other[4] + other[5] * other[6]), 2. * (other[3] * other[5] - other[4] * other[6]), 0., 0., 0., 0.],
                     [2. * (other[3] * other[4] - other[5] * other[6]), 1. - 2. * (other[3]**2 + other[5]**2), 2. * (other[4] * other[5] + other[3] * other[6]), 0., 0., 0., 0.],
                     [2. * (other[3] * other[5] + other[4] * other[6]), 2. * (other[4] * other[5] - other[3] * other[6]), 1. - 2. * (other[3]**2 + other[4]**2), 0., 0., 0., 0.],
                     [0., 0., 0., other[6], other[5], -other[4], -other[3]],
                     [0., 0., 0., -other[5], other[6], other[3], -other[4]],
                     [0., 0., 0., other[4], -other[3], other[6], -other[5]],
                     [0., 0., 0., other[3], other[4], other[5], other[6]]], dtype=np.float64)


def jacobian_self_ominus_other_wrt_other(self, other):
    return np.array([[-1. + 2. * (other[4]**2 + other[5]**2), -2. * (other[3] * other[4] + other[5] * other[6]), -2. * (other[3] * other[5] - other[4] * other[6]), 2. * (other[4] * (self[1] - other[1]) + other[5] * (self[2] - other[2])), 2. * (-2. * other[4] * (self[0] - other[0]) + other[3] * (self[1] - other[1]) - other[6] * (self[2] - other[2])), 2. * (-2. * other[5] * (self[0] - other[0]) + other[6] * (self[1] - other[1]) + other[3] * (self[2] - other[2])), 2. * (other[5] * (self[1] - other[1]) - other[4] * (self[2] - other[2]))],
                     [-2. * (other[3] * other[4] - other[5] * other[6]), -1. + 2. * (other[3]**2 + other[5]**2), -2. * (other[4] * other[5] + other[3] * other[6]), 2. * (other[4] * (self[0] - other[0]) - 2. * other[3] * (self[1] - other[1]) + other[6] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[5] * (self[2] - other[2])), 2. * (-other[6] * (self[0] - other[0]) - 2. * other[5] * (self[1] - other[1]) + other[4] * (self[2] - other[2])), 2. * (-other[5] * (self[0] - other[0]) + other[3] * (self[2] - other[2]))],
                     [-2. * (other[3] * other[5] + other[4] * other[6]), -2. * (other[4] * other[5] - other[3] * other[6]), -1. + 2. * (other[3]**2 + other[4]**2), 2. * (other[5] * (self[0] - other[0]) - other[6] * (self[1] - other[1]) - 2. * other[3] * (self[2] - other[2])), 2. * (other[6] * (self[0] - other[0]) + other[5] * (self[1] - other[1]) - 2. * other[4] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[4] * (self[1] - other[1])), 2. * (other[4] * (self[0] - other[0]) - other[3] * (self[1] - other[1]))],
                     [0., 0., 0., -self[6], -self[5], self[4], self[3]],
                     [0., 0., 0., self[5], -self[6], -self[3], self[4]],
                     [0., 0., 0., -self[4], self[3], -self[6], self[5]],
                     [0., 0., 0., self[3], self[4], self[5], self[6]]], dtype=np.float64)

def jacobian_self_ominus_other_wrt_other_compact(self, other):
    return    np.array([[-1. + 2. * (other[4]**2 + other[5]**2), -2. * (other[3] * other[4] + other[5] * other[6]), -2. * (other[3] * other[5] - other[4] * other[6]), 2. * (other[4] * (self[1] - other[1]) + other[5] * (self[2] - other[2])), 2. * (-2. * other[4] * (self[0] - other[0]) + other[3] * (self[1] - other[1]) - other[6] * (self[2] - other[2])), 2. * (-2. * other[5] * (self[0] - other[0]) + other[6] * (self[1] - other[1]) + other[3] * (self[2] - other[2])), 2. * (other[5] * (self[1] - other[1]) - other[4] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[4] - other[5] * other[6]), -1. + 2. * (other[3]**2 + other[5]**2), -2. * (other[4] * other[5] + other[3] * other[6]), 2. * (other[4] * (self[0] - other[0]) - 2. * other[3] * (self[1] - other[1]) + other[6] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[5] * (self[2] - other[2])), 2. * (-other[6] * (self[0] - other[0]) - 2. * other[5] * (self[1] - other[1]) + other[4] * (self[2] - other[2])), 2. * (-other[5] * (self[0] - other[0]) + other[3] * (self[2] - other[2]))],
                         [-2. * (other[3] * other[5] + other[4] * other[6]), -2. * (other[4] * other[5] - other[3] * other[6]), -1. + 2. * (other[3]**2 + other[4]**2), 2. * (other[5] * (self[0] - other[0]) - other[6] * (self[1] - other[1]) - 2. * other[3] * (self[2] - other[2])), 2. * (other[6] * (self[0] - other[0]) + other[5] * (self[1] - other[1]) - 2. * other[4] * (self[2] - other[2])), 2. * (other[3] * (self[0] - other[0]) + other[4] * (self[1] - other[1])), 2. * (other[4] * (self[0] - other[0]) - other[3] * (self[1] - other[1]))],
                         [0., 0., 0., -self[6], -self[5], self[4], self[3]],
                         [0., 0., 0., self[5], -self[6], -self[3], self[4]],
                         [0., 0., 0., -self[4], self[3], -self[6], self[5]]], dtype=np.float64)

def jacobian_boxplus(self):
    return np.array([[1. - 2. * (self[4]**2 + self[5]**2), 2. * (self[3] * self[4] - self[5] * self[6]), 2. * (self[3] * self[5] + self[4] * self[6]), 0., 0., 0.],
                         [2. * (self[3] * self[4] + self[5] * self[6]), 1. - 2. * (self[3]**2 + self[5]**2), 2. * (self[4] * self[5] - self[3] * self[6]), 0., 0., 0.],
                         [2. * (self[3] * self[5] - self[4] * self[6]), 2. * (self[3] * self[6] + self[4] * self[5]), 1. - 2. * (self[3]**2 + self[4]**2), 0., 0., 0.],
                         [0., 0., 0., self[6], -self[5], self[4]],
                         [0., 0., 0., self[5], self[6], -self[3]],
                         [0., 0., 0., -self[4], self[3], self[6]],
                         [0., 0., 0., -self[3], -self[4], -self[5]]], dtype=np.float64)

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

        r = (self.z - (v1 - v0))[:6]

        J0 = jacobian_self_ominus_other_wrt_other_compact(self.z, sub(v1, v0)) @ jacobian_self_ominus_other_wrt_other(v1, v0) @  jacobian_boxplus(v0)
        J1 = jacobian_self_ominus_other_wrt_other_compact(self.z, sub(v1, v0)) @ jacobian_self_ominus_other_wrt_self(v1, v0) @  jacobian_boxplus(v1)

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

    # draw('before loop-closing', graph)
    graph.solve(step=0)
    # draw('after loop-closing', graph)

    plt.show()
