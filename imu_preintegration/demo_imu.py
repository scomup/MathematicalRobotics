import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
from preintegration import *
import numpy as np
import matplotlib.pyplot as plt
from graph_optimization.plot_pose import *
import quaternion
from imu_factor import *

FILE_PATH = os.path.join(os.path.dirname(__file__), '..')

def getPIM(imu_data, start, end):
    imuIntegrator = imuIntegration(9.81)
    idx = np.where(np.logical_and(imu_data[:,0]>=start, imu_data[:,0]<end))[0]
    for i in idx:
        imuIntegrator.update(imu_data[i,1:4], imu_data[i,4:7], 0.01)
    return imuIntegrator


def draw(figname, gs, color, label):
    fig = plt.figure(figname)
    axes = fig.gca()
    pose_trj = []
    for n in gs.nodes:
        if(not isinstance(n, naviNode)):
            continue
        pose_trj.append(n.state.p)
    pose_trj = np.array(pose_trj)
    axes.scatter(pose_trj[:,0],pose_trj[:,1], c=color, s=10, label=label+' pose')
    imu_trj = []
    for e in gs.edges:
        if(not isinstance(e, imuEdge)):
            continue
        imuIntegrator = imuIntegration(9.80)
        statei = gs.nodes[e.i].state
        biasi = gs.nodes[e.k].bias
        for acc, gyo, dt in zip(e.z.acc_buf, e.z.gyo_buf, e.z.dt_buf):
            imuIntegrator.update(acc, gyo, dt)
            state_new = imuIntegrator.predict(statei,biasi)
            imu_trj.append(state_new.p)
    imu_trj = np.array(imu_trj)
    axes.scatter(imu_trj[:,0],imu_trj[:,1], c=color, s=2,label=label+' imu predict')
    axes.legend()

def draw_bias(figname, gs):
    fig = plt.figure(figname)
    axes = fig.gca()
    bias = []
    for n in gs.nodes:
        if(not isinstance(n, biasNode)):
            continue
        bias.append(n.bias)
    bias = np.array(bias)
    axes.plot(bias[:,0], label='bias acc x')
    axes.plot(bias[:,1], label='bias acc y')
    axes.plot(bias[:,2], label='bias acc z')
    axes.plot(bias[:,3], label='bias gyo x')
    axes.plot(bias[:,4], label='bias gyo y')
    axes.plot(bias[:,5], label='bias gyo z')
    axes.legend()


if __name__ == '__main__':
    imuIntegrator = imuIntegration(9.80)
    pose_file = FILE_PATH+'/data/ndt_pose.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'
    imu_file = FILE_PATH+'/data/imu_data.npy'

    #omegaOdom = np.linalg.inv(np.diag(np.ones(9)*4e-4))
    #omegaMaker = np.linalg.inv(np.diag(np.ones(9)*1e-2))
    omegaBias = np.linalg.inv(np.diag(np.ones(6)*1e-4)) 
    omegaPIM = np.linalg.inv(np.diag(np.array([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1,1,1])))


    pose_data = np.load(pose_file) 
    imu_data = np.load(imu_file)
    gs = graphSolver()
    n= pose_data.shape[0]
    imu_group = []
    begin_idx = 0
    pre_state_idx = 0
    pre_stamp = 0
    for i in range(n):
        p = pose_data[i]
        cur_stamp = p[0]
        if(i == 0):
            state = navState(quaternion.as_rotation_matrix(np.quaternion(*p[4:8])),p[1:4],np.array([0,0,0]))
            pre_state_idx = gs.addNode(naviNode(state))
            pre_bias_idx = gs.addNode(biasNode(np.zeros(6)))
            gs.addEdge(naviEdge(pre_state_idx, state)) 
            gs.addEdge(biasEdge(pre_bias_idx, np.zeros(6), omegaBias))

            pre_stamp = cur_stamp
        else:
            pre_p = gs.nodes[pre_state_idx].state.p
            dt =cur_stamp - pre_stamp
            dist = np.linalg.norm(p[1:4] - pre_p)
            if(dist < 0.1):
                continue
            vel = (p[1:4] - pre_p)/dt
            state = navState(quaternion.as_rotation_matrix(np.quaternion(*p[4:8])),p[1:4],vel)
            cur_state_idx = gs.addNode(naviNode(state)) # add node to graph
            cur_bias_idx = gs.addNode(biasNode(np.zeros(6)))
            gs.addEdge(naviEdge(cur_state_idx, state)) 
            imuIntegrator = getPIM(imu_data, pre_stamp, cur_stamp)
            gs.addEdge(imuEdge(pre_state_idx, cur_state_idx, pre_bias_idx, imuIntegrator,omegaPIM))
            gs.addEdge(biasbetweenEdge(pre_bias_idx, cur_bias_idx, np.eye(6)))
            pre_state_idx = cur_state_idx
            pre_stamp = cur_stamp
    draw('imu pose', gs,'red','before')
    gs.solve()
    draw('imu pose', gs,'green','after')
    plt.grid()
    draw_bias('bias', gs)
    plt.grid()
    plt.show()

