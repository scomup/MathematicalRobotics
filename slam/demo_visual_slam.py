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
    def __init__(self, x, id=0):
        self.x = x
        self.size = x.size
        self.id = id
    def update(self, dx):
        self.x = pose_plus(self.x, dx)


class featureNode:
    def __init__(self, x, id=0):
        self.x = x
        self.size = x.size
        self.id = id
    def update(self, dx):
        self.x = self.x + dx

class reporjEdge:
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
        x = nodes[self.i].x
        p = nodes[self.j].x
        xc2c1, u1, u2, K = self.z
        rl, Jl1, Jl2 = reporj(x, p, u1, K, True)
        rr, Jr1, Jr2 = reporj(pose_plus(xc2c1,x), p, u2, K, True)
        r = np.hstack([rl, rr])
        J1 = np.vstack([Jl1, Jr1])
        J2 = np.vstack([Jl2, Jr2])
        return r, J1, J2
    
def draw3d(figname, frames, points):
    pts = []
    for i in points:
        x = points[i]['p3d']
        pts.append(x)
    pts = np.array(pts)
    for frame in frames:
        x = frame['pose']
        T = makeT(expSO3(x[0:3]),x[3:6])
        plot_pose3(figname, T, 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    axes.scatter(pts[:,0],pts[:,1],pts[:,2])
    set_axes_equal(figname)


def calc_camera_pose(frame, points, x_cw, x_c2c1, K):
    graph = graphSolver()
    idx = graph.addNode(camposeNode(x_cw)) 
    for p in frame['points']:
        if not p in points:
            continue
        u0 = frame['points'][p][0:2]
        u1 = u0.copy()
        u1[0] -= frame['points'][p][2]
        p3d = points[p]['p3d']
        idx_p = graph.addNode(featureNode(p3d),True) 
        #graph.addEdge(reporjEdge(idx, idx_p, [x_c2c1, u0, u1, K]))
        graph.addEdge(reporjEdge(idx, idx_p, [x_c2c1, u0, u1, K],kernel=CauchyKernel(0.5)))
        #r,_,_ = reporjEdge(idx, idx_p, [x_c2c1, u0, u1, K]).residual(graph.nodes)
    graph.solve(False)
    return graph.nodes[idx].x

def initmap(frames, K, x_c2c1):
    baseline = -x_c2c1[3]
    focal = K[0,0]
    Kinv = np.linalg.inv(K)
    points = {}
    for i, frame in enumerate(frames):
        print("initial frame %d..."%i)
        if(i != 0):
            x_cw = calc_camera_pose(frame, points, pose_inv(frames[i-1]['pose']),x_c2c1,K)
            x_wc = pose_inv(x_cw)
            frames[i]['pose'] = x_wc
        for j in list(frame['points']):
            if j in points:
                points[j]['view'].append(i)
                continue
            u,v,disp = frame['points'][j]
            if(disp < 20):
                frame['points'].pop(j)
                continue
            p3d = Kinv.dot(np.array([u,v,1.]))
            depth = (baseline * focal) / (disp)
            p3d *= depth
            p3dw = transform(frames[i]['pose'], p3d)
            points.update({j: {'view':[i],'p3d':p3dw}})
    return points

def remove_outlier(frames, points, K, xc2c1):
    for i, frame in enumerate(frames):
        x = pose_inv(frame['pose'])
        for j in list(frame['points']):
            if j in points:
                pw = points[j]['p3d']
                u0 = frame['points'][j][0:2]
                u1 = u0.copy()
                u1[0] -= frame['points'][j][2]
                u0_reproj = reporj(x, pw, np.zeros(2), K)
                u1_reproj = reporj(pose_plus(xc2c1,x), pw, np.zeros(2), K)
                d0 = np.linalg.norm(u0_reproj - u0)
                d1 = np.linalg.norm(u1_reproj - u1)
                d = d0 + d1
                if(d > 2):
                    del frame['points'][j]
                    idx = points[j]['view'].index(i)
                    points[j]['view'].pop(idx)
                    if(len(points[j]['view']) == 0):
                        del points[j]
                        break





def draw_frame(frames, points, K):
    for frame in frames:
        x = pose_inv(frame['pose'])
        u0s = []
        u1s = []
        for j in frame['points']:
            if j in points:
                pw = points[j]['p3d']
                u0 = frame['points'][j][0:2]
                u1 = reporj(x, pw, np.zeros(2), K)
                u0s.append(u0)
                u1s.append(u1)
        u0s = np.array(u0s)
        u1s = np.array(u1s)
        plt.xlim(0,640)
        plt.ylim(0,400)
        plt.gca().invert_yaxis()
        plt.scatter(u0s[:,0],u0s[:,1])
        plt.scatter(u1s[:,0],u1s[:,1])
        plt.grid()
        plt.show()

def readframes(n,folder):
    frames = []
    for idx in range(0,n):
        fn = folder+'/F%04d.yaml'%idx
        print('read %s...'%fn)
        with open(fn) as file:
            node = yaml.safe_load(file)
            pts = np.array(node['points']['data']).reshape(node['points']['num'],-1)
            pts = dict(zip(pts[:,0].astype(np.int), pts[:,1:].astype(np.float)))
            imus = np.array(node['imu']['data']).reshape(node['imu']['num'],-1)
            frames.append({'stamp':node['stamp'],'pose':np.zeros(6),'points': pts,'imu':imus})
    return frames

def solve(frames, points):
    graph = graphSolver()
    frames_idx = {}
    points_idx = {}
    for i, frame in enumerate(frames):
        x_cw = pose_inv(frame['pose'])
        idx = graph.addNode(camposeNode(x_cw, i)) # add node to graph
        frames_idx.update({i: idx})
    for j in points:
        idx = graph.addNode(featureNode(points[j]['p3d'], j)) # add feature to graph
        points_idx.update({j: idx})
        for i in points[j]['view']:
            f_idx = frames_idx[i]
            u0 = frames[i]['points'][j][0:2]
            u1 = u0.copy()
            u1[0] -= frames[i]['points'][j][2]
            graph.addEdge(reporjEdge(f_idx, idx, [x_c2c1, u0, u1, K],kernel=CauchyKernel(0.1)))      
    graph.report()
    graph.solve()
    graph.report()
    for n in graph.nodes:
        if( type(n).__name__ == 'featureNode'):
            points[n.id]['p3d'] = n.x
        if( type(n).__name__ == 'camposeNode'):
            frames[n.id]['pose'] = pose_inv(n.x)

if __name__ == '__main__':
    fx = 403.5362854003906
    fy = 403.4488830566406
    cx = 323.534423828125
    cy = 203.87405395507812
    x_c2c1 = np.array([0,0,0,-0.075,0,0])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

    frames = readframes(10, 'data/slam')
    points = initmap(frames, K, x_c2c1)
    solve(frames, points)
    remove_outlier(frames, points, K, x_c2c1)
    solve(frames, points)

    #draw_frame(frames, points, K)
    draw3d('view', frames, points)
    plt.show()
