import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from graph_optimization.plot_pose import *

class CamposeVertex:
    def __init__(self, x, id=0):
        self.x = x
        self.size = x.size
        self.id = id
    def update(self, dx):
        self.x = pose_plus(self.x, dx)


class DepthVertex:
    def __init__(self, x, id=0):
        self.x = float(x)
        self.size = 1
        self.id = id
    def update(self, dx):
        self.x = float(self.x + dx)

class DepthEdge:
    def __init__(self, i, z, omega = None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(1)
    def residual(self, vertices):
        depth = vertices[self.i].x
        init_depth = self.z
        return depth - init_depth, np.eye(1)

class Reproj2StereoEdge:
    def __init__(self, i, j, k, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(4)
    def residual(self, vertices):
        x_wbi = vertices[self.i].x
        x_wbj = vertices[self.j].x
        depth = vertices[self.k].x
        p_cj, u_il, u_ir, baseline, K, x_bc = self.z
        rl, Jl1, Jl2, Jl3 = reproj2_stereo(x_wbi, x_wbj, depth, p_cj, u_il, u_ir, baseline, K, x_bc, True)
        return rl, Jl1, Jl2, Jl3


class Reproj2Edge:
    def __init__(self, i, j, k, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(2)
    def residual(self, vertices):
        x_wbi = vertices[self.i].x
        x_wbj = vertices[self.j].x
        depth = vertices[self.k].x
        p_cj, u_il, u_ir, baseline, K, x_bc = self.z        
        rl, Jl1, Jl2, Jl3 = reproj2(x_wbi, x_wbj, depth, p_cj, u_il, K, x_bc, True)
        return rl, Jl1, Jl2, Jl3
    
def draw3d(figname, frames, points, x_bc, R_bi):
    pts = []
    for i in points:
        pc = points[i]['pc'] * points[i]['depth']
        x_wb = frames[points[i]['view'][0]]['pose']
        x_wc = pose_plus(x_wb, x_bc)
        pw = transform(x_wc, pc)
        pts.append(pw)
    pts = np.array(pts)
    for frame in frames:
        x_wb = frame['pose']
        T = tom(x_wb)
        bias = frame['bias'][0:3]
        gravity = transform(x_wb, R_bi.dot(frame['imu'][0][4:7]) - bias)
        plot_pose3_gravity(figname, T, gravity, 0.05)

    fig = plt.figure(figname)
    axes = fig.gca()
    axes.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    set_axes_equal(figname)


def initmap(frames, K, baseline, x_bc, scale):
    print('The map is initialing...')
    focal = K[0, 0]
    Kinv = np.linalg.inv(K)
    points = {}
    for i, frame in enumerate(frames):
        for j in list(frame['points']):
            if j in points:
                points[j]['view'].append(i)
                continue
            u, v, disp = frame['points'][j]
            th = 20/scale
            if (disp < th):
                frame['points'].pop(j)
                continue
            p3d_c = Kinv.dot(np.array([u, v, 1.]))
            depth = (baseline * focal) / (disp)
            points.update({j: {'view':[i], 'pc':p3d_c, 'depth': depth}})
    return points



def draw_frame(frames, points, K, baseline, x_bc, scale):
    for i, frame in enumerate(frames):
        x_wbi = frame['pose']
        u0s = []
        u1s = []
        u0rs = []
        u1rs = []

        for n in frame['points']:
            if n in points and i in points[n]['view']:
                p_cj = points[n]['pc']
                depth = points[n]['depth']
                j = points[n]['view'][0]
                x_wbj = frames[j]['pose']
                # uj_reporj = reproj2(x_wbi, x_wbj, depth, p_cj, np.zeros(2), K, x_bc)
                r = reproj2_stereo(x_wbi, x_wbj, depth, p_cj, np.zeros(2), np.zeros(2), baseline, K, x_bc)
                uj = frame['points'][n][0:2]
                ujr = uj.copy()
                ujr[0] -= frame['points'][n][2]
                u0s.append(r[0:2])
                u1s.append(uj)
                u0rs.append(r[2:4])
                u1rs.append(ujr)

        u0s = np.array(u0s)
        u1s = np.array(u1s)
        u0rs = np.array(u0rs)
        u1rs = np.array(u1rs)

        ab_pairs = np.c_[u0s, u1s]
        ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
        abr_pairs = np.c_[u0rs, u1rs]
        abr_args = abr_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)

        fig, axes = plt.subplots(2, 1, tight_layout=True)
        axes[0].plot(*ab_args, c='k')
        axes[1].plot(*abr_args, c='k')
        axes[0].set_xlim(0, 640/scale)
        axes[0].set_ylim(0, 400/scale)
        axes[1].set_xlim(0, 640/scale)
        axes[1].set_ylim(0, 400/scale)
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
        axes[0].scatter(u0s[:, 0], u0s[:, 1], label = 'reporj')
        axes[0].scatter(u1s[:, 0], u1s[:, 1], label = 'observation')
        axes[1].scatter(u0rs[:, 0], u0rs[:, 1], label = 'reporj')
        axes[1].scatter(u1rs[:, 0], u1rs[:, 1], label = 'observation')

        axes[0].grid()
        axes[1].grid()
        axes[0].legend()
        axes[1].legend()
        plt.show()

def readframes(n, folder, scale):
    frames = []
    for idx in range(0, n):
        fn = folder+'/F%04d.yaml'%idx
        print('read %s...'%fn)
        with open(fn) as file:
            vertex = yaml.safe_load(file)
            pts = np.array(vertex['points']['data']).reshape(vertex['points']['num'], -1)
            pts_d = pts[:, 1:].astype(np.float64)/scale
            pts = dict(zip(pts[:, 0].astype(np.int), pts_d))
            imus = np.array(vertex['imu']['data']).reshape(vertex['imu']['num'], -1)
            frames.append({'stamp':vertex['stamp'], 'pose':np.zeros(6), 'vel':np.zeros(3), 'bias':np.zeros(6), 'points': pts, 'imu':imus})
    return frames


def solve(frames, points, K, baseline, x_bc):
    graph = GraphSolver()
    frames_idx = {}
    points_idx = {}
    for i, frame in enumerate(frames):
        x_wc = frame['pose']
        idx = graph.add_vertex(CamposeVertex(x_wc, i), i==0) # add vertex to graph
        frames_idx.update({i: idx})
    for n in points:
        if (len(points[n]['view'])<2):
            continue
        depth_idx = graph.add_vertex(DepthVertex(np.array([points[n]['depth']]), n), False) # add feature to graph
        graph.add_edge(DepthEdge(depth_idx, np.array([points[n]['depth']]), omega=np.eye(1)))      

        points_idx.update({n: depth_idx})
        bj_idx = frames_idx[points[n]['view'][0]]
        #    reporject a local point in camera j to camera i. # ui
        for i in points[n]['view'][1:]:
            bi_idx = frames_idx[i]
            u_il = frames[i]['points'][n][0:2] 
            u_ir = u_il.copy()
            u_ir[0] -= frames[i]['points'][n][2]
            p_cj = points[n]['pc']
            graph.add_edge(Reproj2StereoEdge(bi_idx, bj_idx, depth_idx, [p_cj, u_il, u_ir, baseline, K, x_bc], kernel=HuberKernel(0.1), omega=reporjOmega))
            # graph.add_edge(Reproj2Edge(bi_idx, bj_idx, depth_idx, [p_cj, u_il, u_ir, baseline, K, x_bc], kernel=HuberKernel(0.5), omega=np.eye(2)*0.01))
    graph.report()
    graph.solve(min_score_change =0.01, step=0)
    graph.report()
    for n in graph.vertices:
        if (type(n).__name__ == 'DepthVertex'):
            points[n.id]['depth'] = n.x
        if (type(n).__name__ == 'CamposeVertex'):
            frames[n.id]['pose'] = n.x

def remove_outlier(frames, points, K, baseline, x_bc):
    for n in points:
        p_cj = points[n]['pc']
        depth = points[n]['depth']
        j = points[n]['view'][0]
        for i in list(points[n]['view'][1:]):
            u_il = frames[i]['points'][n][0:2] 
            u_ir = u_il.copy()
            u_ir[0] -= frames[i]['points'][n][2]
            r = reproj2_stereo(frames[i]['pose'], frames[j]['pose'], depth, p_cj, u_il, u_ir, baseline, K, x_bc, False)
            d = np.linalg.norm(r)
            if d > 1:
                idx = points[n]['view'].index(i)
                points[n]['view'].pop(idx)



if __name__ == '__main__':
    scale = 1.
    fx = 403.5362854003906/scale
    fy = 403.4488830566406/scale
    cx = 323.534423828125/scale
    cy = 203.87405395507812/scale
    baseline = 0.075
    reporjOmega =np.eye(4)*0.01

    x_bc = np.array([-1.20919958,  1.20919958, -1.20919958, 0.0, 0, 0])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    # from imu frame to body frame (# rosrun tf tf_echo oak-d_frame oak_imu_frame)
    R_bi = np.zeros([3, 3])
    t_bi = np.zeros([3])
    R_bi[0, 2] = 1
    R_bi[1, 1] = 1
    R_bi[2, 0] = -1

    if (False):
        frames = readframes(30, 'data/slam', scale)
        points = initmap(frames, K, baseline, x_bc, scale)
        import pickle
        pickle.dump(frames, open("frames.p", "wb"))
        pickle.dump(points, open("points.p", "wb"))
    else:
        import pickle
        frames = pickle.load(open("frames.p", "rb"))
        points = pickle.load(open("points.p", "rb"))

    solve(frames, points, K, baseline, x_bc)
    # remove_outlier(frames, points, K, baseline, x_bc)
    draw3d('view', frames, points, x_bc)
    # draw_frame(frames, points, K, baseline, x_bc, scale)
    plt.show()
