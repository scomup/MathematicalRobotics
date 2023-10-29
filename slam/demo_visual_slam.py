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


class featurevertex:
    def __init__(self, x, id=0):
        self.x = x
        self.size = x.size
        self.id = id

    def update(self, dx):
        self.x = self.x + dx


class reprojEdge:
    def __init__(self, i, j, z, omega=None, kernel=None):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.omega = omega
        self.kernel = kernel
        if (self.omega is None):
            self.omega = np.eye(4)

    def residual(self, vertices):
        x = vertices[self.i].x
        p = vertices[self.j].x
        xc1c2, u1, u2, xbc, K = self.z
        rl, Jl1, Jl2 = reproj(x, p, u1, K, xbc, True)
        rr, Jr1, Jr2 = reproj(x, p, u2, K, pose_plus(xbc, xc1c2), True)
        r = np.hstack([rl, rr])
        J1 = np.vstack([Jl1, Jr1])
        J2 = np.vstack([Jl2, Jr2])
        return r, J1, J2


def draw3d(figname, frames, points, x_bc):
    pts = []
    for i in points:
        x = points[i]['p3d']
        # x = transform(x_bc, x)
        pts.append(x)
    pts = np.array(pts)
    for frame in frames:
        x_wb = frame['pose']
        x_wc = pose_plus(x_wb, x_bc)
        T = tom(x_wb)
        plot_pose3(figname, T, 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    axes.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    set_axes_equal(figname)


def calc_camera_pose(frame, points, x_wc, K, x_c1c2, x_bc):
    graph = GraphSolver()
    idx = graph.add_vertex(CamposeVertex(x_wc))
    for p in frame['points']:
        if p not in points:
            continue
        u0 = frame['points'][p][0:2]
        u1 = u0.copy()
        u1[0] -= frame['points'][p][2]
        p3d = points[p]['p3d']
        idx_p = graph.add_vertex(featurevertex(p3d), True)
        graph.add_edge(reprojEdge(idx, idx_p, [x_c1c2, u0, u1, x_bc, K], kernel=HuberKernel(0.5)))
    graph.solve(False)
    return graph.vertices[idx].x


def initmap(frames, K, x_c1c2, x_bc):
    baseline = 0.075
    focal = K[0, 0]
    Kinv = np.linalg.inv(K)
    points = {}
    for i, frame in enumerate(frames):
        print("initial frame %d..." % i)
        if (i != 0):
            x_wc = calc_camera_pose(frame, points, frames[i-1]['pose'], K, x_c1c2, x_bc)
            frames[i]['pose'] = x_wc
        for j in list(frame['points']):
            if j in points:
                points[j]['view'].append(i)
                continue
            u, v, disp = frame['points'][j]
            if (disp < 20/640.):
                frame['points'].pop(j)
                continue
            p3d_c = Kinv.dot(np.array([u, v, 1.]))
            depth = (baseline * focal) / (disp)
            p3d_c *= depth
            p3d_b = transform(x_bc, p3d_c)
            p3d_w = transform(frames[i]['pose'], p3d_b)
            points.update({j: {'view': [i], 'p3d': p3d_w}})
            # u1 = reproj(frames[i]['pose'], p3d_w, np.array([u, v]), K, x_bc, calcJ = False)
            # u2 = reproj(frames[i]['pose'], p3d_w, np.array([u-disp, v]), K, pose_plus(x_bc, x_c1c2), calcJ = False)
            # print(u1, u2)
    return points


def remove_outlier(frames, points, K, x_c1c2, x_bc):
    for i, frame in enumerate(frames):
        x = frame['pose']
        for j in list(frame['points']):
            if j in points:
                pw = points[j]['p3d']
                u0 = frame['points'][j][0:2]
                u1 = u0.copy()
                u1[0] -= frame['points'][j][2]
                u0_reproj = reproj(x, pw, np.zeros(2), K, x_bc)  # x, p, u1, K, xbc, True
                u1_reproj = reproj(x, pw, np.zeros(2), K, pose_plus(x_bc, x_c1c2))
                d0 = np.linalg.norm(u0_reproj - u0)
                d1 = np.linalg.norm(u1_reproj - u1)
                d = d0 + d1
                if (d0 > 2/640.):
                    del frame['points'][j]
                    idx = points[j]['view'].index(i)
                    points[j]['view'].pop(idx)
                    if (len(points[j]['view']) == 0):
                        del points[j]
                        break


def draw_frame(frames, points, K, x_c1c2, x_bc):
    for frame in frames:
        x_wb = frame['pose']
        u0s = []
        u1s = []
        for j in frame['points']:
            if j in points:
                pw = points[j]['p3d']
                u0 = frame['points'][j][0:2]
                u1 = reproj(x_wb, pw, np.zeros(2), K, x_bc)
                u0s.append(u0)
                u1s.append(u1)
        u0s = np.array(u0s)
        u1s = np.array(u1s)
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.5, 0.5)
        plt.gca().invert_yaxis()
        plt.scatter(u0s[:, 0], u0s[:, 1])
        plt.scatter(u1s[:, 0], u1s[:, 1])
        plt.grid()
        plt.show()


def readframes(n, folder, W, H):
    frames = []
    for idx in range(0, n):
        fn = folder+'/F%04d.yaml' % idx
        print('read %s...' % fn)
        with open(fn) as file:
            vertex = yaml.safe_load(file)
            pts = np.array(vertex['points']['data']).reshape(vertex['points']['num'], -1)
            pts_d = pts[:, 1:].astype(np.float64)
            pts_d[:, 0] /= W
            pts_d[:, 1] /= H
            pts_d[:, 0:2] -= 0.5
            pts_d[:, 2] /= W
            pts = dict(zip(pts[:, 0].astype(np.int32), pts_d))
            imus = np.array(vertex['imu']['data']).reshape(vertex['imu']['num'], -1)
            frames.append({'stamp': vertex['stamp'],
                           'pose': np.zeros(6),
                           'vel': np.zeros(3),
                           'bias': np.zeros(6),
                           'points': pts,
                           'imu': imus})
    return frames


def solve(frames, points, K, x_c1c2, x_bc):
    graph = GraphSolver()
    frames_idx = {}
    points_idx = {}
    for i, frame in enumerate(frames):
        x_wc = frame['pose']
        idx = graph.add_vertex(CamposeVertex(x_wc, i))  # add vertex to graph
        frames_idx.update({i: idx})
    for j in points:
        idx = graph.add_vertex(featurevertex(points[j]['p3d'], j))  # add feature to graph
        # idx = graph.add_vertex(featurevertex(np.array([1., 0., 0.]), j)) # add feature to graph
        points_idx.update({j: idx})
        for i in points[j]['view']:
            f_idx = frames_idx[i]
            u0 = frames[i]['points'][j][0:2]
            u1 = u0.copy()
            u1[0] -= frames[i]['points'][j][2]
            graph.add_edge(reprojEdge(f_idx, idx, [x_c1c2, u0, u1, x_bc, K], kernel=HuberKernel(0.1)))
    graph.report()
    graph.solve(step=1)
    graph.report()
    for n in graph.vertices:
        if (type(n).__name__ == 'featurevertex'):
            points[n.id]['p3d'] = n.x
        if (type(n).__name__ == 'CamposeVertex'):
            frames[n.id]['pose'] = n.x

if __name__ == '__main__':
    W = 640.
    H = 400.
    fx = 403.5362854003906/W
    fy = 403.4488830566406/H
    cx = 323.534423828125/W - 0.5
    cy = 203.87405395507812/H - 0.5
    x_c1c2 = np.array([0, 0, 0, 0.075, 0, 0])
    x_bc = np.array([-1.20919958,  1.20919958, -1.20919958, 0.0, 0, 0])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])

    frames = readframes(2, 'data/slam', W, H)
    points = initmap(frames, K, x_c1c2, x_bc)
    solve(frames, points, K, x_c1c2, x_bc)
    # remove_outlier(frames, points, K, x_c1c2)
    # solve(frames, points, K, x_c1c2, x_bc)
    draw3d('view', frames, points, x_bc)
    # draw_frame(frames, points, K, x_c1c2, x_bc)
    plt.show()
