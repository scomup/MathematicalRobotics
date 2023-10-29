import numpy as np
from graph_solver import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from utilities.polygon import *
from matplotlib.patches import Polygon
from graph_optimization.demo_pose2d_graph import pose2dEdge, Pose2dbetweenEdge


class Pose2polygonEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if (self.omega is None):
            self.omega = np.eye(3)

    def residual(self, vertices):
        x = vertices[self.i].x[0:2]
        r = polygonRes(x, self.z)
        res = np.zeros(3)
        J = np.zeros([3, 3])
        J[0:2, 0:2] = numericalDerivative(polygonRes, [x, self.z], 0)
        res[0:2] = r
        return res, J

class Pose2Vertex:
    def __init__(self, x):
        self.x = x
        self.size = x.size

    def update(self, dx):
        self.x = m2v(v2m(self.x).dot(v2m(dx)))


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    for n in graph.vertices:
        plot_pose2(figname, v2m(n.x), 0.2)
    for e in graph.edges:
        if (e.type == 'one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt(v2m(graph.vertices[i].x))
        _, tj = makeRt(v2m(graph.vertices[j].x))
        x = [ti[0], tj[0]]
        y = [ti[1], tj[1]]
        axes.plot(x, y, c=e.color, linestyle=':')


def draw_polygon(figname, ploygons):
    fig = plt.figure(figname)
    axes = fig.gca()
    axes.cla()
    for poly in ploygons:
        polygon = Polygon(poly)
        axes.add_patch(polygon)

if __name__ == '__main__':
    ploygons = []

    ploygons.append(np.array([[-5, 3], [5, 3], [5, 1], [0.5, 1], [0, 0], [-0.5, 1], [-5., 1]]))
    ploygons.append(np.array([[-5, -1], [3.5, -1], [4, 0], [4.5, -1], [5, -1], [5, -3], [-5., -3]]))
    # ploygons.append(np.array([[7, -3], [10, -3], [10, 5], [8, 5]]))

    graph = GraphSolver()
    n = 27

    cur_pose = np.array([-6, -0.5, 0])
    odom = np.array([0.5, 0, 0.01])
    odomOmega = np.diag([10, 10, 1.])
    polyOmega = np.diag([2, 2, 0.])
    poseOmega = np.diag([1000, 1000, 0.])
    for i in range(n):
        graph.add_vertex(Pose2Vertex(cur_pose))  # add vertex to graph
        if (i == 0):
            graph.add_edge(pose2dEdge(i, cur_pose, poseOmega))
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))
    graph.add_edge(pose2dEdge(n - 1, cur_pose, poseOmega))

    for i in range(n-1):
        j = (i + 1)
        graph.add_edge(Pose2dbetweenEdge(i, j, odom, odomOmega))  # add edge(i, j) to graph

    for i in range(n):
        for poly in ploygons:
            graph.add_edge(Pose2polygonEdge(i, poly, polyOmega))  #

    draw_polygon('before loop-closing', ploygons)
    draw('before loop-closing', graph)

    graph.report()
    # graph.solve()
    step = 1.
    iter = 0
    last_score = None
    while(True):
        dx, score = graph.solve_once()
        iter += 1
        print('iter %d: %f' % (iter, score))
        # if (np.linalg.norm(dx)>step):
        #    dx = dx/np.linalg.norm(dx) * step
        graph.update(dx)
        if (last_score is None):
            last_score = score
            continue
        if (last_score < score):
            break
        if (last_score - score < 0.0001):
            break
        last_score = score

        draw_polygon('after loop-closing', ploygons)
        draw('after loop-closing', graph)
        plt.pause(0.1)
    graph.report()
    plt.show()
