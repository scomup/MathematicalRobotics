import numpy as np
from graph_solver import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from utilities.plot_tools import *
from graph_optimization.demo_pose3d_graph import Pose3dEdge, Pose3dbetweenEdge, Pose3Vertex

FILE_PATH = os.path.join(os.path.dirname(__file__), '..')


def draw(figname, graph):
    for n in graph.vertices:
        plot_pose3(figname, expSE3(n.x), 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    for e in graph.edges:
        if (e.type == 'one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt(expSE3(graph.vertices[i].x))
        _, tj = makeRt(expSE3(graph.vertices[j].x))
        x = [ti[0], tj[0]]
        y = [ti[1], tj[1]]
        z = [ti[2], tj[2]]
        axes.plot(x, y, z, c=e.color, linestyle=':')
    set_axes_equal(figname)


def to2d(x):
    R = expSO3(x[0:3])
    theta = np.arctan2(R[1, 0], R[0, 0])
    x2d = np.zeros(3)
    x2d[0:2] = x[3:5]
    x2d[2] = theta
    return x2d


def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    pose_trj = []
    for n in graph.vertices:
        if (not isinstance(n, Pose3Vertex)):
            continue
        pose_trj.append(makeRt(n.x)[1])
    pose_trj = np.array(pose_trj)
    axes.scatter(pose_trj[:, 0], pose_trj[:, 1], c='blue', s=2, label='ndt optimize')
    e1_trj = []
    e2_trj = []
    for e in graph.edges:
        if (not isinstance(e, Pose3dEdge)):
            continue
        t = makeRt(graph.vertices[e.link[0]].x)[1]
        e1_trj.append(t)
        t = makeRt(e.z)[1]
        e2_trj.append(t)

    e1_trj = np.array(e1_trj)
    e2_trj = np.array(e2_trj)
    axes.scatter(e1_trj[:, 0], e1_trj[:, 1], c='blue', s=10, label='maker(ndt)')
    axes.scatter(e2_trj[:, 0], e2_trj[:, 1], c='red', s=10, label='maker(truth)')


def draw_graph_pose(c, l):
    pose_trj = []
    for n in graph.vertices:
        if (not isinstance(n, Pose3Vertex)):
            continue
        pose_trj.append(makeRt(expSE3(n.x))[1])
    pose_trj = np.array(pose_trj)
    plt.scatter(pose_trj[:, 0], pose_trj[:, 1], c=c, s=2, label=l)


def find_nearest(data, stamp):
    idx = (np.abs(data[:, 0] - stamp)).argmin()
    return data[idx, :]

if __name__ == '__main__':
    pose_file = FILE_PATH+'/data/ndt_pose.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'
    pose_data = np.load(pose_file)
    truth_data = np.load(truth_file)
    graph = GraphSolver()

    T0 = None
    T0_idx = 0
    last_marker_T = None
    omegaOdom = np.linalg.inv(np.diag(np.ones(6)*4e-4))
    omegaMaker = np.linalg.inv(np.diag(np.ones(6)*1e-2))
    mark_dist = 5
    truth_trj = []
    for p in pose_data:
        T1 = makeT(quaternion_to_matrix(p[4:8]), p[1:4])
        truth_trj.append(find_nearest(truth_data, p[0])[1:4])
        if (T0 is None):
            T0 = T1
            last_marker_T = T1
            T0_idx = graph.add_vertex(Pose3Vertex(T0))  # add vertex to graph
            continue
        # pt = find_nearest(truth_data, p[0])
        # T1t = makeT(quaternion_to_matrix(pt[4:8])), pt[1:4])

        T1_idx = graph.add_vertex(Pose3Vertex(T1))
        delta = np.linalg.inv(T0).dot(T1)
        # print(T0)
        graph.add_edge(Pose3dbetweenEdge([T0_idx, T1_idx], delta, omegaOdom))
        T0_idx = T1_idx
        T0 = T1

        if (np.linalg.norm(makeRt(np.linalg.inv(last_marker_T) @ T1)[1]) > mark_dist):
            last_marker_T = T1
            marker = find_nearest(truth_data, p[0])
            marker_T = makeT(quaternion_to_matrix(marker[4:8]), marker[1:4])
            # marker_T[0:3, 3] += np.random.normal(0, 0.01, 3)
            graph.add_edge(Pose3dEdge([T1_idx], marker_T, omegaMaker))  # add prior pose to graph

    # draw_graph_pose('blue', 'before ndt')

    graph.solve()
    draw(1, graph)
    plt.plot(truth_data[:, 1], truth_data[:, 2], label='truth', color='m')
    plt.plot(pose_data[:, 1], pose_data[:, 2], label='ndt old', color='cyan')
    # draw_graph_pose('red', 'after ndt')

    aft_trj = []
    for n in graph.vertices:
        if (not isinstance(n, Pose3Vertex)):
            continue
        aft_trj.append(makeRt(n.x)[1])
    aft_trj = np.array(aft_trj)
    truth_trj = np.array(truth_trj)
    err = np.linalg.norm((truth_trj - aft_trj), axis=1)
    avg_err = np.average(err)
    worst_err = np.max(err)
    print("err: %f" % avg_err)
    print("worst err: %f" % worst_err)
    # plt.scatter(aft_trj[:, 0], aft_trj[:, 1], label='aft_trj', color='m', s=5)
    # plt.scatter(truth_trj[:, 0], truth_trj[:, 1], label='truth_trj', color='cyan', s=5)

    plt.grid()
    plt.legend()
    plt.show()
