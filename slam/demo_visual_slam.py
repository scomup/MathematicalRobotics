import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from graph_optimization.plot_pose import *


class camposeNode:
    def __init__(self, x):
        self.x = x
        self.size = x.size
    def update(self, dx):
        self.x = pose_plus(self.x, dx)

class featureNode:
    def __init__(self, x):
        self.x = x
        self.size = x.size
    def update(self, dx):
        self.x = self.x + dx

class priorposeEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(6)
    def residual(self, nodes):
        x = nodes[self.i].x
        R = expSO3(x[0:3])
        t = x[3:6]
        Rz = expSO3(self.z[0:3])
        tz = self.z[3:6]
        dR = R.T.dot(R)
        dt = R.T.dot(tz - t)
        r = np.hstack([logSO3(dR),dt])
        J = -np.eye(6)
        J[0:3,0:3] = -dR.T
        J[3:6,0:3] = skew(dt)
        return r, J


class reporjEdge:
    def __init__(self, i, j, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(2)

    def residual(self, nodes):
        x = nodes[self.i].x
        p = nodes[self.j].x
        pim, K = self.z
        r, J1, J2 = reporj(x, p, pim, K, True)
        return r, J1, J2
    

class pointEdge:
    def __init__(self, i, z, omega = None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(2)

    def residual(self, nodes):
        p = nodes[self.i].x
        x, pim, K = self.z
        r, _, J = reporj(x, p, pim, K, True)
        return r, J

class stereoPointEdge:
    def __init__(self, i, z, omega = None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(4)
    def residual(self, nodes):
        p = nodes[self.i].x
        x, xc2c1, u0, u1, K = self.z
        r0, _, J0 = reporj(x, p, u0, K, True)
        r1, _, J1 = reporj(pose_plus(xc2c1,x), p, u1, K, True)
        r = np.hstack([r0,r1])
        J = np.vstack([J0,J1])
        return r, J


def draw(figname, frames_pose, points):
    for x in frames_pose:
        T = makeT(expSO3(x[0:3]),x[3:6])
        plot_pose3(figname, T, 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    axes.scatter(points[:,0],points[:,1],points[:,2])
    set_axes_equal(figname)

class camposeEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(2)
    def residual(self, nodes):
        x = nodes[self.i].x
        p, pim, K = self.z
        r, J1, _ = reporj(x, p, pim, K, True)
        return r, J1


def calc_camera_pose(frame, points):
    graph = graphSolver()
    idx = graph.addNode(camposeNode(np.zeros(6))) 
    for p in frame['points']:
        if not p in points:
            continue
        pim = frame['points'][p][0:2].astype(float)
        p3d = points[p]['p3d']
        graph.addEdge(camposeEdge(idx, [p3d, pim, K]))
    graph.solve(False)
    return graph.nodes[idx].x

def inv(x):
    R = expSO3(x[0:3])
    t = x[3:6]
    Rinv = np.linalg.inv(R)
    return np.hstack([logSO3(Rinv),Rinv.dot(-t)])

def tom(x):    
    R = expSO3(x[0:3])
    t = x[3:6]
    return makeT(R,t)

def tox(m):    
    R,t = makeRt(m)
    return np.hstack([logSO3(R),t])

def init(frames, K):
    baseline = 0.075
    focal = K[0,0]
    Kinv = np.linalg.inv(K)
    frames_pose = [np.zeros(6)]
    points = {}
    for i, frame in enumerate(frames):
        if(len(frames_pose)<=i):
            x_cw = calc_camera_pose(frame, points)
            x_wc = inv(x_cw)
            frames_pose.append(x_wc)
        for j in frame['points']:
            if j in points:
                points[j]['view'].append(i)
                continue
            u,v,disp = frame['points'][j]
            p3d = Kinv.dot(np.array([u,v,1.]))
            depth = (baseline * focal) / (disp)
            p3d *= depth
            p3dw = transform(frames_pose[i], p3d)
            points.update({j: {'view':[i],'p3d':p3dw}})
    return frames_pose, points


if __name__ == '__main__':
    frames = []
    points = {}
    fx = 403.5362854003906
    fy = 403.4488830566406
    cx = 323.534423828125
    cy = 203.87405395507812
    x_c2c1 = np.array([0,0,0,-0.075,0,0])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

    n = 10
    for idx in range(n):
        with open('data/slam/F%04d.yaml'%idx) as file:
            node = yaml.safe_load(file)
            pts = np.array(node['points']['data']).reshape(node['points']['num'],-1)
            imus = np.array(node['imu']['data']).reshape(node['imu']['num'],-1)
            frames.append({'stamp':node['stamp'],'points': dict(zip(pts[:,0].astype(np.int), pts[:,1:])),'imu':imus})
    frame_pose, points = init(frames, K)

    points_idx = {}
    graph = graphSolver()
    for j in points:
        if(len(points[j]['view'])<=5):
            continue
        idx = graph.addNode(featureNode(points[j]['p3d'])) # add feature to graph
        points_idx.update({j: idx})
        for i in points[j]['view']:
            x = inv(frame_pose[i])
            u0 = frames[i]['points'][j][0:2].astype(float)
            u1 = frames[i]['points'][j][0:2].astype(float)
            u1[0] -= frames[i]['points'][j][2].astype(float)
            #graph.addEdge(pointEdge(idx, [x, pim, K], kernel=CauchyKernel(0.1)))
            graph.addEdge(stereoPointEdge(idx, [x, x_c2c1, u0, u1, K],kernel=CauchyKernel(0.5)))
            r, J = stereoPointEdge(idx, [x, x_c2c1, u0, u1, K]).residual(graph.nodes)
    
    graph.report()
    graph.solve()
    graph.report()

    pts = []
    for n in graph.nodes:
        pts.append(n.x)
    pts = np.array(pts)


    draw('view',frame_pose, pts)
    plt.show()


    exit(0)
    

    """
    graph = graphSolver()
    frames_idx = {}
    points_idx = {}
    for i, frame in enumerate(frames):
        x_cw = inv(frame_pose[i])
        idx = graph.addNode(camposeNode(x_cw)) # add node to graph
        #graph.addEdge(priorposeEdge(idx, x_cw, np.eye(6)*10000.))
        frames_idx.update({i: idx})
    for i in points:
        view = points[i]['view']
        p_idx = graph.addNode(featureNode(points[i]['p3d'])) # add feature to graph
        points_idx.update({i: p_idx})
        for f in view:
            f_idx =  frames_idx[f]
            pim = frames[f]['points'][i][0:2].astype(float)
            graph.addEdge(reporjEdge(f_idx, p_idx, [pim, K], kernel=CauchyKernel(0.1)))
    graph.report()
    graph.solve()
    graph.report()
    """
    
