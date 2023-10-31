import numpy as np
from graph_solver import *
from g2o_io import *
from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex
import matplotlib.pyplot as plt
from graph_optimization.plot_pose import *


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.add_subplot(projection='3d')
    vertices = []
    edges = []
    for v in graph.vertices:
        vertices.append((v.x)[0:3, 3])
    for e in graph.edges:
        edges.append([*(graph.vertices[e.i].x)[0:3, 3]])
        edges.append([*(graph.vertices[e.j].x)[0:3, 3]])
    vertices = np.array(vertices)
    axes.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],  s=10, color='r')
    edges = np.array(edges)
    axes.plot(xs=edges[:, 0], ys=edges[:, 1], zs=edges[:, 2], c='b', linewidth=1)
    set_axes_equal(figname)


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

    draw('before loop-closing', graph)
    graph.solve(step=0)
    draw('after loop-closing', graph)

    plt.show()
