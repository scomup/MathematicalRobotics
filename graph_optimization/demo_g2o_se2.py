import numpy as np
from graph_solver import *
from g2o_io import *
from graph_optimization.demo_pose2d_graph import Pose2dEdge, Pose2dbetweenEdge, Pose2Vertex, draw
import matplotlib.pyplot as plt


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    vertices = []
    edges = []
    for v in graph.vertices:
        vertices.append(v.x)
    for e in graph.edges:
        edges.append([*graph.vertices[e.i].x[:2], *graph.vertices[e.j].x[:2]])
    vertices = np.array(vertices)
    edges = np.array(edges)
    axes.scatter(vertices[:,0], vertices[:,1], s=2, color='k')
    ab_pairs = edges
    plt_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
    axes.plot(*plt_args, c='b', linewidth=1)


if __name__ == '__main__':

    graph = GraphSolver(use_sparse = True)
    edges, vertices = load_g2o_se2('data/g2o/manhattanOlson3500.g2o')

    for vertex in vertices:
        graph.add_vertex(Pose2Vertex(vertex[1]))  # add vertex to graph

    for edge in edges:
        graph.add_edge(Pose2dbetweenEdge(edge[0][0], edge[0][1], edge[1], edge[2]))  # add edge(i, j) to graph

    draw('before loop-closing', graph)
    graph.solve()
    draw('after loop-closing', graph)

    plt.show()

