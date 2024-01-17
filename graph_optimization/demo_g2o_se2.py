import numpy as np
from graph_solver import *
from utilities.g2o_io import *
from graph_optimization.demo_pose2d_graph import Pose2dEdge, Pose2dbetweenEdge, Pose2Vertex, draw
import matplotlib.pyplot as plt


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    vertices = []
    edges_odom = []
    edges_loop = []
    for v in graph.vertices:
        vertices.append(v.x[0:2, 2])
    for e in graph.edges:
        if(len(e.link) != 2):
            continue
        i, j = e.link
        if(np.abs(i-j) == 1):
            edges_odom.append([*graph.vertices[i].x[0:2, 2]])
            edges_odom.append([*graph.vertices[j].x[0:2, 2]])
        else:
            edges_loop.append([*graph.vertices[i].x[0:2, 2]])
            edges_loop.append([*graph.vertices[j].x[0:2, 2]])
    vertices = np.array(vertices)
    edges_odom = np.array(edges_odom)
    edges_loop = np.array(edges_loop)
    axes.scatter(vertices[:, 0], vertices[:, 1], s=2, color='k')
    axes.plot(edges_odom[:, 0], edges_odom[:, 1], c='b', linewidth=1)
    axes.plot(edges_loop[:, 0], edges_loop[:, 1], c='r', linewidth=1)

if __name__ == '__main__':

    graph = GraphSolver(use_sparse=True)
    # kernel = GaussianKernel(5)
    # intel.g2o
    # manhattanOlson3500.g2o
    # ringCity.g2o
    path = os.path.dirname(os.path.abspath(__file__))

    edges, vertices = load_g2o_se2(path+'/../data/g2o/manhattanOlson3500.g2o')

    for vertex in vertices:
        if(vertex[0] == 0):
            graph.add_vertex(Pose2Vertex(vertex[1]), is_constant=True)
        else:
            graph.add_vertex(Pose2Vertex(vertex[1]))  # add vertex to graph

    for edge in edges:
        graph.add_edge(Pose2dbetweenEdge(edge[0], edge[1], edge[2]))  # add edge(i, j) to graph

    draw('before loop-closing', graph)
    graph.solve()
    draw('after loop-closing', graph)

    plt.show()
