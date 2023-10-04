import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from imu_preintegration.preintegration import *
from imu_preintegration.imu_factor import *
from slam.demo_visual_slam2 import *
from graph_optimization.plot_pose import *

G = 9.81


class gravityEdge:
    def __init__(self, i, z, omega=None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(3)

    def residual(self, nodes):
        x_wb = nodes[self.i].x.vec()[0:6]
        pim = self.z
        acc_b = pim.acc_buf[0]
        acc_w, J, _ = transform(x_wb, acc_b, True)
        Jt = np.zeros([3, 9])
        Jt[:, 0:6] = J
        return acc_w - np.array([0, 0, G]), Jt

class gravityBiasEdge:
    def __init__(self, i, j, z, omega=None, kernel=None):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(3)

    def residual(self, nodes):
        x_wb = nodes[self.i].x.vec()[0:6]
        bias = nodes[self.j].x[0:3]
        pim = self.z
        acc_b = pim.acc_buf[0]
        acc_w, J1, J2 = transform(x_wb, acc_b - bias, True)
        J1t = np.zeros([3, 9])
        J1t[:, 0:6] = J1
        J2t = np.zeros([3, 6])
        J2t[:, 0:3] = -J2
        return acc_w - np.array([0, 0, 9.81]), J1t, J2t


class reproj2StereoEdge:
    def __init__(self, i, j, k, z, omega=None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(4)

    def residual(self, nodes):
        x_wbi = nodes[self.i].x.vec()[0:6]
        x_wbj = nodes[self.j].x.vec()[0:6]
        depth = nodes[self.k].x
        p_cj, u_il, u_ir, baseline, K, x_bc = self.z
        rl, J1, J2, J3 = reproj2_stereo(x_wbi, x_wbj, depth, p_cj, u_il, u_ir, baseline, K, x_bc, True)
        J1t = np.zeros([4, 9])
        J2t = np.zeros([4, 9])
        J1t[:, 0:6] = J1
        J2t[:, 0:6] = J2
        return rl, J1t, J2t, J3


class reproj2Edge:
    def __init__(self, i, j, k, z, omega=None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(2)

    def residual(self, nodes):
        x_wbi = nodes[self.i].x.vec()[0:6]
        x_wbj = nodes[self.j].x.vec()[0:6]
        depth = nodes[self.k].x
        p_cj, u_il, u_ir, baseline, K, x_bc = self.z
        rl, J1, J2, J3 = reproj2(x_wbi, x_wbj, depth, p_cj, u_il, K, x_bc, True)
        J1t = np.zeros([2, 9])
        J2t = np.zeros([2, 9])
        J1t[:, 0:6] = J1
        J2t[:, 0:6] = J2

        return rl, J1t, J2t, J3


def getPIM(frame, R_bi, t_bi):
    imuIntegrator = imuIntegration(G, Rbi=R_bi, tbi=t_bi)
    for imu_data in frame['imu']:
        imuIntegrator.update(imu_data[4:7], imu_data[1:4], 1/400.)
    return imuIntegrator

def solve(frames, points, K, baseline, x_bc, use_imu=True):
    graph = graphSolver()
    frames_idx = {}
    """
    Add frame and bias nodes
    """
    bias_idx = {}
    for i, frame in enumerate(frames):
        x_wb = frame['pose']
        vel = frame['vel']
        bias = frame['bias']
        state = navState.set(np.hstack([x_wb, vel]))
        # add node to graph
        idx = graph.addNode(naviNode(state, frame['stamp'], i))
        frames_idx.update({i: idx})
        idxb = graph.addNode(biasNode(bias, i))
        bias_idx.update({i: idxb})
    """
    Add frame and bias nodes
    """
    graph.addEdge(biasEdge(bias_idx[0], np.zeros(6), biasOmega))
    graph.addEdge(naviEdge(frames_idx[0], graph.nodes[frames_idx[0]].x, prirOmega))
    """
    Add imu edges
    """
    if (use_imu):
        for j in range(1, len(frames)):
            i = j - 1
            idx_i = frames_idx[i]
            idx_j = frames_idx[j]
            idxb_i = bias_idx[i]
            idxb_j = bias_idx[j]
            imuIntegrator = getPIM(frames[i], R_bi, t_bi)
            # add imu preintegration to graph
            graph.addEdge(imupreintEdge(idx_i, idx_j, idxb_i, imuIntegrator, imupreintOmega))
            # add the relationship between velocity and position to graph
            graph.addEdge(posvelEdge(idx_i, idx_j, imuIntegrator.d_tij, posvelOmega))
            # add the bias change error to graph
            graph.addEdge(biaschangeEdge(idxb_i, idxb_j, biaschangeOmega))
            graph.addEdge(gravityEdge(idx_i, imuIntegrator, gravityOmega))
            # graph.addEdge(gravityBiasEdge(idx_i, idxb_i, imuIntegrator, np.eye(3)*0.1))
    """
    Add reproj edges
    """
    points_idx = {}
    for n in points:
        if (len(points[n]['view']) < 2):
            continue
        depth_idx = graph.addNode(depthNode(np.array([points[n]['depth']]), n), False)
        graph.addEdge(depthEdge(depth_idx, np.array([points[n]['depth']]), omega=np.eye(1)))
        points_idx.update({n: depth_idx})
        bj_idx = frames_idx[points[n]['view'][0]]
        for i in points[n]['view'][1:]:
            bi_idx = frames_idx[i]
            u_il = frames[i]['points'][n][0:2]
            u_ir = u_il.copy()
            u_ir[0] -= frames[i]['points'][n][2]
            p_cj = points[n]['pc']
            graph.addEdge(reproj2StereoEdge(bi_idx, bj_idx, depth_idx,
                                            [p_cj, u_il, u_ir, baseline, K, x_bc],
                                            kernel=CauchyKernel(0.1), omega=reporjOmega))

    graph.report()
    graph.solve(min_score_change=0.01, step=0)
    graph.report()
    for n in graph.nodes:
        if (type(n).__name__ == 'depthNode'):
            points[n.id]['depth'] = n.x
        if (type(n).__name__ == 'naviNode'):
            v = n.x.vec()
            frames[n.id]['pose'] = v[0:6]
            frames[n.id]['vel'] = v[6:9]
        if (type(n).__name__ == 'biasNode'):
            frames[n.id]['bias'] = n.x


def drawBias(frames):
    bias = []
    for i, frame in enumerate(frames):
        bias.append(frame['bias'])
    bias = np.array(bias)
    plt.plot(bias[:, 0], label="acc x")
    plt.plot(bias[:, 1], label="acc y")
    plt.plot(bias[:, 2], label="acc z")
    plt.plot(bias[:, 3], label="gyo x")
    plt.plot(bias[:, 4], label="gyo y")
    plt.plot(bias[:, 5], label="gyo z")
    plt.legend()
    plt.show()

def drawVel(frames):
    vel = []
    for i, frame in enumerate(frames):
        vel.append(frame['vel'])
    vel = np.array(vel)
    plt.plot(vel[:, 0], label="vel x")
    plt.plot(vel[:, 1], label="vel y")
    plt.plot(vel[:, 2], label="vel z")
    plt.legend()
    plt.show()

def drawG(frames, R_bi):
    acc = []
    for i, frame in enumerate(frames):
        x_wb = frame['pose']
        bias = frame['bias'][0:3]
        for a in frame['imu']:
            acc_corrected = R_bi.dot(a[4:7]) - bias
            acc.append(transform(x_wb, acc_corrected))
    acc = np.array(acc)
    plt.plot(acc[:, 0], label="acc x")
    plt.plot(acc[:, 1], label="acc y")
    plt.plot(acc[:, 2], label="acc z")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    scale = 1.
    fx = 403.5362854003906/scale
    fy = 403.4488830566406/scale
    cx = 323.534423828125/scale
    cy = 203.87405395507812/scale
    baseline = 0.075
    # from camera frame to body frame (rosrun tf tf_echo oak-d_frame oak_left_camera_optical_frame)
    x_bc = np.array([-1.20919958,  1.20919958, -1.20919958, 0.0, 0, 0])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    # from imu frame to body frame (# rosrun tf tf_echo oak-d_frame oak_imu_frame)
    R_bi = np.zeros([3, 3])
    t_bi = np.zeros([3])
    R_bi[0, 2] = 1
    R_bi[1, 1] = 1
    R_bi[2, 0] = -1

    imupreintOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    biaschangeOmega = np.linalg.inv(np.diag(np.ones(6)*1e-1))
    biasOmega = np.linalg.inv(np.diag(np.ones(6)*1e2))
    posvelOmega = np.linalg.inv(np.diag(np.ones(3)*1e-3))
    reporjOmega = np.eye(4)*0.01
    prirOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    gravityOmega = np.eye(3)*0.1
    prirOmega[0:3, 0:3] = 0
    navitransformOmega = np.linalg.inv(np.diag(np.ones(9)*1e-4))
    navitransformOmega[6:9, 6:9] = 0

    if (True):
        frames = readframes(10, 'data/slam', scale)
        points = initmap(frames, K, baseline, x_bc, scale)
        import pickle
        pickle.dump(frames, open("frames.p", "wb"))
        pickle.dump(points, open("points.p", "wb"))
    else:
        import pickle
        frames = pickle.load(open("frames.p", "rb"))
        points = pickle.load(open("points.p", "rb"))

    solve(frames, points, K, baseline, x_bc, True)
    remove_outlier(frames, points, K, baseline, x_bc)
    solve(frames, points, K, baseline, x_bc)

    draw3d('view', frames, points, x_bc, R_bi)
    # draw_frame(frames, points, K, baseline, x_bc, scale)
    # drawBias(frames)
    # drawVel(frames)
    # drawG(frames, R_bi)
    plt.show()
