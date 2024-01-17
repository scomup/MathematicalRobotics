import numpy as np
import matplotlib.pyplot as plt
from graph_solver import *
from utilities.g2o_io import *
from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex
from utilities.plot_tools import *


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.add_subplot(projection='3d')
    vertices = []
    edges_odom = []
    edges_loop = []
    for v in graph.vertices:
        vertices.append(v.x[0:3, 3])
    for e in graph.edges:
        if(len(e.link) != 2):
            continue
        i, j = e.link
        if(np.abs(i-j) == 1):
            edges_odom.append([*graph.vertices[i].x[0:3, 3]])
            edges_odom.append([*graph.vertices[j].x[0:3, 3]])
        else:
            edges_loop.append([*graph.vertices[i].x[0:3, 3]])
            edges_loop.append([*graph.vertices[j].x[0:3, 3]])
    vertices = np.array(vertices)
    edges_odom = np.array(edges_odom)
    edges_loop = np.array(edges_loop)
    axes.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],  s=5, color='k')
    axes.plot(xs=edges_odom[:, 0], ys=edges_odom[:, 1], zs=edges_odom[:, 2], c='b', linewidth=1)
    axes.plot(xs=edges_loop[:, 0], ys=edges_loop[:, 1], zs=edges_loop[:, 2], c='r', linewidth=1)
    set_axes_equal(figname)


if __name__ == '__main__':

    graph = GraphSolver(use_sparse=True)
    # parking-garage.g2o
    # sphere2500.g2o
    path = os.path.dirname(os.path.abspath(__file__))
    edges, vertices = load_g2o_se3(path+'/../data/g2o/sphere2500.g2o')
    kernel = None
    # kernel = HuberKernel(1)
    for vertex in vertices:
        if(vertex[0] == 0):
            graph.add_vertex(Pose3Vertex(vertex[1]), is_constant=True)
        else:
            graph.add_vertex(Pose3Vertex(vertex[1]))  # add vertex to graph
    for edge in edges:
        graph.add_edge(Pose3dbetweenEdge(edge[0], edge[1], edge[2], kernel=kernel))

    draw('before loop-closing', graph)
    graph.solve(step=0)
    draw('after loop-closing', graph)

    plt.show()
