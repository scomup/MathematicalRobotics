import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from imu_preintegration.preintegration import *
from imu_preintegration.imu_factor import *
from slam.demo_visual_slam import *
from graph_optimization.plot_pose import *


class reporj2Edge:
    def __init__(self, i, j, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(4)
    def residual(self, nodes):
        x = nodes[self.i].x.vec()[0:6]
        p = nodes[self.j].x
        xc1c2, u1, u2, xbc, K = self.z
        rl, Jl1, Jl2 = reporj(x, p, u1, K, xbc, True)
        rr, Jr1, Jr2 = reporj(x, p, u2, K, pose_plus(xbc,xc1c2), True)
        r = np.hstack([rl, rr])
        J1 = np.vstack([Jl1, Jr1])
        J1t = np.zeros([4,9])
        J1t[:,0:6] = J1
        J2 = np.vstack([Jl2, Jr2])
        return r, J1t, J2

def getPIM(frame, R_bi, t_bi):

    imuIntegrator = imuIntegration(9.81, Rbi = R_bi, tbi = t_bi)
    for imu_data in frame['imu']:
        imuIntegrator.update(imu_data[4:7], imu_data[1:4], 1/400.)
    return imuIntegrator


def solve2(frames, points, K, x_c1c2, x_bc, R_bi, t_bi, fix_points=False, fix_state=False):
    graph = graphSolver()
    frames_idx = {}
    points_idx = {}
    bias_idx = {}
    for i, frame in enumerate(frames):
        x_wc = frame['pose']
        vel = frame['vel']
        bias = frame['bias']
        state = navState.set(np.hstack([x_wc,vel]))
        idx = graph.addNode(naviNode(state, frame['stamp'],i),fix_state) # add node to graph
        frames_idx.update({i: idx})
        idxb = graph.addNode(biasNode(bias, i))
        bias_idx.update({i: idxb})
        if(i != 0):
            imuIntegrator = getPIM(frame, R_bi, t_bi)
            graph.addEdge(imupreintEdge(frames_idx[i-1], frames_idx[i], bias_idx[i-1], imuIntegrator, imupreintOmega)) # add imu preintegration to graph
            graph.addEdge(posvelEdge(frames_idx[i-1], frames_idx[i], imuIntegrator.d_tij, posvelOmega)) # add the relationship between velocity and position to graph
            graph.addEdge(biaschangeEdge(bias_idx[i-1], bias_idx[i], biaschangeOmega)) # add the bias change error to graph
        else:
            graph.addEdge(biasEdge(bias_idx[0], np.zeros(6), biasOmega))
            graph.addEdge(naviEdge(frames_idx[0], state, prirOmega))
       
    for j in points:
        idx = graph.addNode(featureNode(points[j]['p3d'], j),fix_points) # add feature to graph
        points_idx.update({j: idx})
        for i in points[j]['view']:
            f_idx = frames_idx[i]
            u0 = frames[i]['points'][j][0:2]
            u1 = u0.copy()
            u1[0] -= frames[i]['points'][j][2]
            graph.addEdge(reporj2Edge(f_idx, idx, [x_c1c2, u0, u1, x_bc, K], reporjOmega,kernel=CauchyKernel(0.05)))   #kernel=CauchyKernel(0.5)
    graph.report()
    graph.solve(min_step= 0.01, step=1)
    graph.report()
    graph.edges[2].z.predict(graph.nodes[0].x,graph.nodes[1].x)
    for n in graph.nodes:
        if( type(n).__name__ == 'featureNode'):
            points[n.id]['p3d'] = n.x
        if( type(n).__name__ == 'naviNode'):
            v = n.x.vec()
            frames[n.id]['pose'] = v[0:6]
            frames[n.id]['vel'] = v[6:9]
        if( type(n).__name__ == 'biasNode'):
            frames[n.id]['bias'] = n.x

def drawBias(frames):
    bias = []
    for i, frame in enumerate(frames):
        bias.append(frame['bias'])
    bias = np.array(bias)  
    plt.plot(bias[:,0],label="acc x")
    plt.plot(bias[:,1],label="acc y")
    plt.plot(bias[:,2],label="acc z")
    plt.plot(bias[:,3],label="gyo x")
    plt.plot(bias[:,4],label="gyo y")
    plt.plot(bias[:,5],label="gyo z")
    plt.legend()
    plt.show()

def drawVel(frames):
    vel = []
    for i, frame in enumerate(frames):
        vel.append(frame['vel'])
    vel = np.array(vel)  
    plt.plot(vel[:,0],label="vel x")
    plt.plot(vel[:,1],label="vel y")
    plt.plot(vel[:,2],label="vel z")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    W = 640.
    H = 400.
    fx = 403.5362854003906/W
    fy = 403.4488830566406/H
    cx = 323.534423828125/W - 0.5
    cy = 203.87405395507812/H - 0.5
    #fx = 403.5362854003906
    #fy = 403.4488830566406
    #cx = 323.534423828125
    #cy = 203.87405395507812

    x_c1c2 = np.array([0,0,0,0.075,0,0])
    x_bc = np.array([-1.20919958,  1.20919958, -1.20919958,0.0,0,0])
    K = np.array([[fx,0, cx],[0, fy,cy],[0,0,1.]])

    imupreintOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    biaschangeOmega = np.linalg.inv(np.diag(np.ones(6)*1e-1)) 
    biasOmega = np.linalg.inv(np.diag(np.ones(6)*1e2))
    posvelOmega = np.linalg.inv(np.diag(np.ones(3)*1e-3))
    reporjOmega = np.linalg.inv(np.diag(np.ones(4)*1e-2))
    prirOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    prirOmega[0:3,0:3] = 0

    frames = readframes(20, 'data/slam', W, H)
    points = initmap(frames, K, x_c1c2, x_bc)
    
    R_bi = np.zeros([3,3])
    t_bi = np.zeros([3])
    R_bi[0,2] = 1
    R_bi[1,1] = -1
    R_bi[2,0] = -1
    #solve(frames, points, K, x_c1c2, x_bc)
    solve2(frames, points, K, x_c1c2, x_bc, R_bi, t_bi, fix_points=True, fix_state=False)
    solve2(frames, points, K, x_c1c2, x_bc, R_bi, t_bi, fix_points=False, fix_state=True)
    #draw3d('view', frames, points, x_bc)
    #draw_frame(frames, points, K, x_c1c2, x_bc)
    #drawBias(frames)
    drawVel(frames)
    plt.show()
