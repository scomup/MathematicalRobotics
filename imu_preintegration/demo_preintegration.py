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


def draw(axes, graph, color, label):
    pose_trj = []
    for n in graph.nodes:
        if(not isinstance(n, naviNode)):
            continue
        pose_trj.append(n.state.p)
    pose_trj = np.array(pose_trj)
    axes.scatter(pose_trj[:,0],pose_trj[:,1], c=color, s=10, label=label+' pose')
    imu_trj = []
    for e in graph.edges:
        if(not isinstance(e, imupreintEdge)):
            continue
        imuIntegrator = imuIntegration(9.80)
        statei = graph.nodes[e.i].state
        biasi = graph.nodes[e.k].bias
        for acc, gyo, dt in zip(e.z.acc_buf, e.z.gyo_buf, e.z.dt_buf):
            imuIntegrator.update(acc, gyo, dt)
            state_new = imuIntegrator.predict(statei,biasi)
            imu_trj.append(state_new.p)
    imu_trj = np.array(imu_trj)
    axes.scatter(imu_trj[:,0],imu_trj[:,1], c=color, s=2,label=label+' imu predict')
    #axes.legend()

def draw_bias(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    bias = []
    for n in graph.nodes:
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
def draw_vel(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    vel = []
    for n in graph.nodes:
        if(not isinstance(n, naviNode)):
            continue
        vel.append(n.state.v)
    vel = np.array(vel)
    axes.plot(vel[:,0], label='vel x')
    axes.plot(vel[:,1], label='vel y')
    axes.plot(vel[:,2], label='vel z')
    axes.legend()

def print_error(graph, truth_data):
    aft_trj = []
    truth_trj = []
    for n in graph.nodes:
        if(not isinstance(n, naviNode)):
            continue
        aft_trj.append(n.state.p)
        truth_p = find_nearest(truth_data, n.stamp)
        truth_trj.append(truth_p[1:4])
    aft_trj = np.array(aft_trj)
    truth_trj  = np.array(truth_trj)
    err = np.linalg.norm((truth_trj - aft_trj),axis=1)
    avg_err = np.average(err)
    worst_err = np.max(err)
    print("avg err:%f"%avg_err)
    print("worst err:%f"%worst_err)
    return err

if __name__ == '__main__':
    imuIntegrator = imuIntegration(9.80)
    pose_file = FILE_PATH+'/data/ndt_pose.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'
    imu_file = FILE_PATH+'/data/imu_data.npy'

    navitransformOmega = np.linalg.inv(np.diag(np.ones(9)*1e-1))
    navitransformOmega[6:9,6:9] = 0
    makeromega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    makeromega[0:3,0:3] = 0
    makeromega[6:9,6:9] = 0
    biasOmega = np.linalg.inv(np.diag(np.ones(6)*1e-2))
    biaschangeOmega = np.linalg.inv(np.diag(np.ones(6)*1e-4)) 
    imupreintOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    omegaNdtPose = np.linalg.inv(np.diag(np.array([1,1,1,1,1,1,100,100,100])))
    posvelOmega = np.linalg.inv(np.diag(np.ones(3)*1e1))


    pose_data = np.load(pose_file) 
    imu_data = np.load(imu_file)
    truth_data = np.load(truth_file)
    graph = graphSolver()
    
    mark_dist = 10
    
    for i, p in enumerate(pose_data):
        cur_stamp = p[0]
        if(i == 0):
            state = navState(quaternion.as_rotation_matrix(np.quaternion(*p[4:8])),p[1:4],np.array([0,0,0]))
            pre_state_idx = graph.addNode(naviNode(state, cur_stamp))
            pre_bias_idx = graph.addNode(biasNode(np.zeros(6)))
            graph.addEdge(biasEdge(pre_bias_idx, np.zeros(6), biasOmega))
            pre_state_idx = 0
        else:
            pre_p = graph.nodes[pre_state_idx].state.p
            dt = cur_stamp - graph.nodes[pre_state_idx].stamp
            dist = np.linalg.norm(p[1:4] - pre_p)
            #if(dist < 0.1):
            #    continue
            vel = (p[1:4] - pre_p)/dt
            state = navState(quaternion.as_rotation_matrix(np.quaternion(*p[4:8])),p[1:4],vel)
            cur_state_idx = graph.addNode(naviNode(state, cur_stamp)) # add first naviState to graph
            cur_bias_idx = graph.addNode(biasNode(np.zeros(6)))


            imuIntegrator = getPIM(imu_data, graph.nodes[pre_state_idx].stamp, cur_stamp)

            delta = graph.nodes[pre_state_idx].state.local(state,False)
            graph.addEdge(navitransEdge(pre_state_idx, cur_state_idx, delta, navitransformOmega))
            graph.addEdge(imupreintEdge(pre_state_idx, cur_state_idx, pre_bias_idx, imuIntegrator,imupreintOmega)) # add imu preintegration to graph
            graph.addEdge(posvelEdge(pre_state_idx, cur_state_idx, imuIntegrator.d_tij, posvelOmega)) # add the relationship between velocity and position to graph
            graph.addEdge(biaschangeEdge(pre_bias_idx, cur_bias_idx, biaschangeOmega)) # add the bias change error to graph
            pre_state_idx = cur_state_idx
    
    marker_list = []
    last_pose = np.array([0,0,0])
    for idx, n in enumerate(graph.nodes):
        if not isinstance(n, naviNode):
            continue
        if(np.linalg.norm(last_pose - n.state.p)>mark_dist or idx == 0 or idx >= len(graph.nodes)-2):
            last_pose = n.state.p
            marker = find_nearest(truth_data, n.stamp)
            marker = navState(quaternion.as_rotation_matrix(np.quaternion(*marker[4:8])),marker[1:4],np.array([0,0,0]))
            #marker.p += np.random.normal(0, 0.01, 3)
            graph.addEdge(naviEdge(idx, marker, makeromega))
            marker_list.append(marker.p)
            #break
    
      
    marker_list = np.array(marker_list)
    fig = plt.figure('imu pose')
    axes = fig.gca()
    #err = print_error(graph, truth_data)
    draw(axes, graph,'red','before')
    graph.report()
    graph.solve()
    graph.report()
    draw(axes, graph,'green','after')
    axes.plot(truth_data[:,1],truth_data[:,2],label='truth', color='blue')
    axes.scatter(marker_list[:,0],marker_list[:,1],label='marker',s=50,color='black')
    axes.grid()
    axes.legend()
    err = print_error(graph, truth_data)
    fig = plt.figure('error ')
    axes = fig.gca()
    axes.plot(err)
    axes.axhline(0.012, color='red', lw=1, alpha=0.7) #
    axes.axhline(0.036, color='red', lw=1, alpha=0.7) #
    draw_bias('bias', graph)
    draw_vel('vel', graph)
    plt.grid()
    plt.show()


