
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
from preintegration import *
import numpy as np
import matplotlib.pyplot as plt
from utilities.plot_tools import *
from imu_factor import *


FILE_PATH = os.path.join(os.path.dirname(__file__), '..')


def draw(figname, graph, color, label):
    fig = plt.figure(figname)
    axes = fig.gca()
    pose_trj = []
    for n in graph.vertices:
        if (not isinstance(n, NaviVertex)):
            continue
        pose_trj.append(n.x.p)
    pose_trj = np.array(pose_trj)
    axes.scatter(pose_trj[:, 0], pose_trj[:, 1], c=color, s=10, label=label)


def find_nearest(data, stamp):
    idx = (np.abs(data[:, 0] - stamp)).argmin()
    return data[idx, :]


def print_error(truth_trj):
    aft_trj = []
    for n in graph.vertices:
        if (not isinstance(n, NaviVertex)):
            continue
        aft_trj.append(n.x.p)
    aft_trj = np.array(aft_trj)
    truth_trj = np.array(truth_trj)
    err = np.linalg.norm((truth_trj - aft_trj), axis=1)
    avg_err = np.average(err)
    worst_err = np.max(err)
    print("avg err:%f" % avg_err)
    print("worst err:%f" % worst_err)

if __name__ == '__main__':
    pose_file = FILE_PATH+'/data/ndt_pose.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'
    pose_data = np.load(pose_file)
    truth_data = np.load(truth_file)
    graph = GraphSolver(True)

    state0 = None
    state0_idx = 0
    last_marker = None

    omegaOdom = np.linalg.inv(np.diag(np.ones(9)*4e-4))
    omegaMaker = np.linalg.inv(np.diag(np.ones(9)*1e-2))
    mark_dist = 1
    truth_trj = []

    for p in pose_data:
        state1 = NavState(quaternion_to_matrix(np.roll(p[4:8], -1)), p[1:4], np.array([0, 0, 0]))
        truth_trj.append(find_nearest(truth_data, p[0])[1:4])
        if (state0 is None):
            state0 = state1
            last_marker = state1
            state0_idx = graph.add_vertex(NaviVertex(state1))  # add vertex to graph
            continue

        state1_idx = graph.add_vertex(NaviVertex(state1))
        delta = state0.local(state1, False)
        graph.add_edge(NavitransEdge([state0_idx, state1_idx], delta, omegaOdom))
        state0_idx = state1_idx
        state0 = state1

        if (np.linalg.norm(last_marker.local(state1).p) > mark_dist):
            last_marker = state1
            marker = find_nearest(truth_data, p[0])
            marker = NavState(quaternion_to_matrix(np.roll(marker[4:8], -1)), marker[1:4], np.array([0, 0, 0]))
            marker.p += np.random.normal(0, 0.01, 3)
            graph.add_edge(NaviEdge([state1_idx], marker, omegaMaker))  # add prior pose to graph

    graph.report()
    print_error(truth_trj)
    draw(1, graph, 'red', ' ndt pose before')
    graph.solve()
    draw(1, graph, 'blue', 'ndt pose after')
    plt.plot(truth_data[:, 1], truth_data[:, 2], label='truth', color='green')
    plt.legend()
    plt.grid()
    plt.legend()
    plt.show()
    print_error(truth_trj)
    graph.report()


