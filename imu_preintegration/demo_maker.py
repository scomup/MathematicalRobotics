
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
from preintegration import *
import numpy as np
import matplotlib.pyplot as plt
from graph_optimization.plot_pose import *
from imu_factor import *

import quaternion
FILE_PATH = os.path.join(os.path.dirname(__file__), '..')


def draw(figname, gs, color, label):
    fig = plt.figure(figname)
    axes = fig.gca()
    pose_trj = []
    for n in gs.nodes:
        if(not isinstance(n, naviNode)):
            continue
        pose_trj.append(n.state.p)
    pose_trj = np.array(pose_trj)
    axes.scatter(pose_trj[:,0],pose_trj[:,1], c=color, s=10, label=label)

def find_nearest(data, stamp):
    idx = (np.abs(data[:,0] - stamp)).argmin()
    return data[idx,:]


if __name__ == '__main__':
    
    pose_file = FILE_PATH+'/data/ndt_pose.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'
    pose_data = np.load(pose_file) 
    truth_data = np.load(truth_file)
    gs = graphSolver()
    
    state0 = None
    state0_idx = 0
    last_marker = None

    omegaOdom = np.linalg.inv(np.diag(np.ones(9)*4e-4))
    omegaMaker = np.linalg.inv(np.diag(np.ones(9)*1e-2)) 
    mark_dist = 5
    truth_trj = []

    for p in pose_data:
        state1 = navState(quaternion.as_rotation_matrix(np.quaternion(*p[4:8])),p[1:4],np.array([0,0,0]))
        truth_trj.append(find_nearest(truth_data, p[0])[1:4])
        if(state0 is None):
            state0 = state1
            last_marker = state1
            state0_idx = gs.addNode(naviNode(state1)) # add node to graph
            continue

        state1_idx = gs.addNode(naviNode(state1))
        delta = state0.local(state1,False)
        gs.addEdge(navibetweenEdge(state0_idx, state1_idx, delta, omegaOdom))
        state0_idx = state1_idx
        state0 = state1

        if( np.linalg.norm(last_marker.local(state1).p)>mark_dist):
            last_marker = state1
            marker = find_nearest(truth_data, p[0])
            marker = navState(quaternion.as_rotation_matrix(np.quaternion(*marker[4:8])),marker[1:4],np.array([0,0,0]))
            #marker_T[0:3,3] += np.random.normal(0, 0.01, 3)
            gs.addEdge(naviEdge(state1_idx, marker, omegaMaker)) # add prior pose to graph
    draw(1,gs,'red','ndt pose before')
    gs.solve()
    draw(1,gs,'blue','ndt pose after')
    plt.plot(truth_data[:,1],truth_data[:,2],label='truth', color='green')
    plt.grid()
    plt.legend()
    plt.show()


