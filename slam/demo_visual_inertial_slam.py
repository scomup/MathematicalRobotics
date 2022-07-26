import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from graph_optimization.plot_pose import *
from imu_preintegration.preintegration import *
from imu_preintegration.imu_factor import *
from slam.demo_visual_slam import *

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
        xc2c1, u1, u2, K = self.z
        rl, Jl1, Jl2 = reporj(x, p, u1, K, True)
        rr, Jr1, Jr2 = reporj(pose_plus(xc2c1,x), p, u2, K, True)
        r = np.hstack([rl, rr])
        J1 = np.vstack([Jl1, Jr1])
        J1t = np.zeros([4,9])
        J1t[:,0:6] = J1
        J2 = np.vstack([Jl2, Jr2])
        return r, J1t, J2

def getPIM(frame, xbi):
    imuIntegrator = imuIntegration(9.81, Tbi = xbi)
    for imu_data in frame['imu']:
        imuIntegrator.update(imu_data[1:4], imu_data[4:7], 1/400.)
    return imuIntegrator


def solve2(frames, points, xbi):
    graph = graphSolver()
    frames_idx = {}
    points_idx = {}
    bias_idx = {}
    for i, frame in enumerate(frames):
        x_cw = pose_inv(frame['pose'])
        state = navState.set(np.hstack([x_cw,np.zeros(3)]))
        idx = graph.addNode(naviNode(state, frame['stamp'],i)) # add node to graph
        frames_idx.update({i: idx})
        """
        idxb = graph.addNode(biasNode(np.zeros(6)))
        bias_idx.update({i: idxb})
        if(i != 0):
            imuIntegrator = getPIM(frame, xbi)
            graph.addEdge(imupreintEdge(frames_idx[i-1], frames_idx[i], bias_idx[i-1], imuIntegrator, imupreintOmega)) # add imu preintegration to graph
            #graph.addEdge(posvelEdge(pre_state_idx, cur_state_idx, imuIntegrator.d_tij, posvelOmega)) # add the relationship between velocity and position to graph
            graph.addEdge(biaschangeEdge(bias_idx[i-1], bias_idx[i], biaschangeOmega)) # add the bias change error to graph
        else:
            graph.addEdge(biasEdge(bias_idx[0], np.zeros(6), biasOmega))
        """

    for j in points:
        idx = graph.addNode(featureNode(points[j]['p3d'], j),True) # add feature to graph
        points_idx.update({j: idx})
        for i in points[j]['view']:
            f_idx = frames_idx[i]
            u0 = frames[i]['points'][j][0:2]
            u1 = u0.copy()
            u1[0] -= frames[i]['points'][j][2]
            graph.addEdge(reporj2Edge(f_idx, idx, [x_c2c1, u0, u1, K],kernel=CauchyKernel(0.1)))      
    graph.report()
    graph.solve()
    graph.report()
    for n in graph.nodes:
        if( type(n).__name__ == 'featureNode'):
            points[n.id]['p3d'] = n.x
        if( type(n).__name__ == 'naviNode'):
            frames[n.id]['pose'] = n.x.vec()[0:6]

if __name__ == '__main__':
    fx = 403.5362854003906
    fy = 403.4488830566406
    cx = 323.534423828125
    cy = 203.87405395507812
    imupreintOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    biaschangeOmega = np.linalg.inv(np.diag(np.ones(6)*1e-4)) 
    biasOmega = np.linalg.inv(np.diag(np.ones(6)*1e-2))
    x_c2c1 = np.array([0,0,0,-0.075,0,0])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

    frames = readframes(3, 'data/slam')
    points = initmap(frames, K, x_c2c1)
    
    xbi = np.array([ 0.,  0., -1.57079633, -0.014,  0.053, 0.])
    #xbi = np.array([ 0, 0, 0, 0, 0, 0.])

    solve2(frames, points, xbi)
    #remove_outlier(frames, points, K, x_c2c1)
    #solve2(frames, points, xbi)
    draw3d('view', frames, points)
    plt.show()
