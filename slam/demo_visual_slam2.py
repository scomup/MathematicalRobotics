import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from graph_optimization.plot_pose import *
USE_SCALE = False
class camposeNode:
    def __init__(self, x, id=0):
        self.x = x
        self.size = x.size
        self.id = id
    def update(self, dx):
        self.x = pose_plus(self.x, dx)


class depthNode:
    def __init__(self, x, id=0):
        self.x = x
        self.size = 1
        self.id = id
    def update(self, dx):
        self.x = self.x + dx

class depthEdge:
    def __init__(self, i, z, omega = None, kernel=None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(1)
    def residual(self, nodes):
        depth = nodes[self.i].x
        init_depth = self.z
        return depth - init_depth, np.eye(1)

class reproj2StereoEdge:
    def __init__(self, i, j, k, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(4)
    def residual(self, nodes):
        x_wbi = nodes[self.i].x
        x_wbj = nodes[self.j].x
        depth = nodes[self.k].x
        p_cj, u_il, u_ir, x_crcl, K, x_bc = self.z
        rl, Jl1, Jl2, Jl3 = reproj2_stereo(x_wbi, x_wbj, depth, p_cj, u_il, u_ir, x_crcl, K, x_bc, True)
        return rl, Jl1, Jl2, Jl3


class reproj2Edge:
    def __init__(self, i, j, k, z, omega = None, kernel=None):
        self.i = i
        self.j = j
        self.k = k
        self.z = z
        self.type = 'three'
        self.omega = omega
        self.kernel = kernel
        if(self.omega is None):
            self.omega = np.eye(2)
    def residual(self, nodes):
        x_wbi = nodes[self.i].x
        x_wbj = nodes[self.j].x
        depth = nodes[self.k].x
        p_cj, u_il, u_ir, x_crcl, K, x_bc = self.z        
        rl, Jl1, Jl2, Jl3 = reproj2(x_wbi, x_wbj, depth, p_cj, u_il, K, x_bc, True)
        return rl, Jl1, Jl2, Jl3
    
def draw3d(figname, frames, points, x_bc):
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
        plot_pose3(figname, T, 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    axes.scatter(pts[:,0],pts[:,1],pts[:,2])
    set_axes_equal(figname)


def initmap(frames, K, x_c1c2, x_bc):
    print('The map is initialing...')
    baseline = 0.075
    focal = K[0,0]
    Kinv = np.linalg.inv(K)
    points = {}
    for i, frame in enumerate(frames):
        for j in list(frame['points']):
            if j in points:
                points[j]['view'].append(i)
                continue
            u,v,disp = frame['points'][j]
            th = 20
            if(USE_SCALE):
                th = 20/640.
            if(disp < th):
                frame['points'].pop(j)
                continue
            p3d_c = Kinv.dot(np.array([u,v,1.]))
            depth = (baseline * focal) / (disp)
            points.update({j: {'view':[i],'pc':p3d_c, 'depth': depth,'dd': 0}})
    return points



def draw_frame(frames, points, K, x_c1c2, x_bc):
    for i, frame in enumerate(frames):
        x_wbi = frame['pose']
        u0s = []
        u1s = []
        #bad0 = []
        #bad1 = []

        for n in frame['points']:
            if n in points:
                p_cj = points[n]['pc']
                depth = points[n]['depth']
                j = points[n]['view'][0]
                #x_wbi = frames[i]['pose']
                x_wbj = frames[j]['pose']
                uj_reporj = reproj2(x_wbi, x_wbj, depth, p_cj, np.zeros(2), K, x_bc)
                uj = frame['points'][n][0:2]
                u0s.append(uj_reporj)
                u1s.append(uj)
                #if(points[n]['dd'] > 0.5):
                #if(n in [161, 238, 273, 353, 374, 388, 353, 417, 528]):
                #    bad0.append(uj_reporj)#161 238 273 353 374 388 353 417 528
                #    bad1.append(uj)
        u0s = np.array(u0s)
        u1s = np.array(u1s)
        ab_pairs = np.c_[u0s, u1s]
        ab_args = ab_pairs.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)

        # segments
        plt.plot(*ab_args, c='k')

        #bad0 = np.array(bad0)
        #bad1 = np.array(bad1)
        if(USE_SCALE):
            plt.xlim(-0.5,0.5)
            plt.ylim(-0.5,0.5)
        else:
            plt.xlim(0,640)
            plt.ylim(0,400)

        plt.gca().invert_yaxis()
        plt.scatter(u0s[:,0],u0s[:,1])
        plt.scatter(u1s[:,0],u1s[:,1])
        #if(bad0.shape[0] > 0):
        #    plt.scatter(bad0[:,0],bad0[:,1], c = 'red')
        #    plt.scatter(bad1[:,0],bad1[:,1], c = 'black')
        plt.grid()
        plt.show()

def readframes(n,folder,W,H):
    frames = []
    for idx in range(0,n):
        fn = folder+'/F%04d.yaml'%idx
        print('read %s...'%fn)
        with open(fn) as file:
            node = yaml.safe_load(file)
            pts = np.array(node['points']['data']).reshape(node['points']['num'],-1)
            pts_d = pts[:,1:].astype(np.float)
            if(USE_SCALE):
                pts_d[:,0] /= W
                pts_d[:,1] /= H
                pts_d[:,0:2] -= 0.5
                pts_d[:,2] /= W
            pts = dict(zip(pts[:,0].astype(np.int), pts_d))
            imus = np.array(node['imu']['data']).reshape(node['imu']['num'],-1)
            frames.append({'stamp':node['stamp'],'pose':np.zeros(6),'vel':np.zeros(3),'bias':np.zeros(6),'points': pts,'imu':imus})
    return frames


def solve(frames, points, K, x_c1c2, x_bc):
    graph = graphSolver()
    frames_idx = {}
    points_idx = {}
    x_crcl = pose_inv(x_c1c2)
    for i, frame in enumerate(frames):
        x_wc = frame['pose']
        idx = graph.addNode(camposeNode(x_wc, i),i==0) # add node to graph
        frames_idx.update({i: idx})
    for n in points:
        if(len(points[n]['view'])<2):
            continue
        depth_idx = graph.addNode(depthNode(np.array([points[n]['depth']]), n),False) # add feature to graph
        graph.addEdge(depthEdge(depth_idx, np.array([points[n]['depth']]),omega=np.eye(1)))      

        points_idx.update({n: depth_idx})
        bj_idx = frames_idx[points[n]['view'][0]]
        #    reporject a local point in camera j to camera i. #ui
        for i in points[n]['view'][1:]:
            bi_idx = frames_idx[i]
            u_il = frames[i]['points'][n][0:2] 
            u_ir = u_il.copy()
            u_ir[0] -= frames[i]['points'][n][2]
            p_cj = points[n]['pc']
            graph.addEdge(reproj2StereoEdge(bi_idx, bj_idx, depth_idx, [p_cj, u_il, u_ir, x_crcl, K, x_bc],kernel=HuberKernel(0.5),omega=np.eye(4)*0.01))
            #graph.addEdge(reproj2Edge(bi_idx, bj_idx, depth_idx, [p_cj, u_il, u_ir, x_crcl, K, x_bc],kernel=HuberKernel(0.1)))
    graph.report()
    graph.solve(min_score_change =0.01, step=0)
    graph.report()
    #depths = []
    for n in graph.nodes:
        if( type(n).__name__ == 'depthNode'):
            #depths.append([np.abs(points[n.id]['depth']-float(n.x)), n.id])
            points[n.id]['dd'] = np.abs(points[n.id]['depth']-float(n.x))
            points[n.id]['depth'] = n.x
        if( type(n).__name__ == 'camposeNode'):
            frames[n.id]['pose'] = n.x
    #depths = np.array(depths)
    #dict(zip(pts[:,0].astype(np.int), pts_d))
    #depths = depths[depths[:, 0].argsort()]
    #plt.plot(depths[:,0], label = 'old')
    #plt.plot(depths[:,1], label = 'new')
    #plt.legend()
    #plt.show()
    #return depths

def remove_outlier(frames, points, K, x_c1c2, x_bc):
    x_crcl = pose_inv(x_c1c2)
    for n in points:
        p_cj = points[n]['pc']
        depth = points[n]['depth']
        j = points[n]['view'][0]
        for i in list(points[n]['view'][1:]):
            u_il = frames[i]['points'][n][0:2] 
            u_ir = u_il.copy()
            u_ir[0] -= frames[i]['points'][n][2]
            r = reproj2_stereo(frames[i]['pose'], frames[j]['pose'], depth, p_cj, u_il, u_ir, x_crcl, K, x_bc, False)
            d = np.linalg.norm(r)
            if d > 1:
                idx = points[n]['view'].index(i)
                points[n]['view'].pop(idx)



if __name__ == '__main__':
    W = 640.
    H = 400.
    fx = 403.5362854003906/W
    fy = 403.4488830566406/H
    cx = 323.534423828125/W - 0.5
    cy = 203.87405395507812/H - 0.5
    if(not USE_SCALE):
        fx = 403.5362854003906
        fy = 403.4488830566406
        cx = 323.534423828125
        cy = 203.87405395507812
    x_c1c2 = np.array([0,0,0,0.075,0,0])
    x_bc = np.array([-1.20919958,  1.20919958, -1.20919958,0.0,0,0])
    K = np.array([[fx,0, cx],[0, fy,cy],[0,0,1.]])

    frames = readframes(10, 'data/slam',W,H)
    points = initmap(frames, K, x_c1c2, x_bc)
    solve(frames, points, K, x_c1c2, x_bc)
    remove_outlier(frames, points, K, x_c1c2, x_bc)
    #draw3d('view',frames, points, x_bc)
    draw_frame(frames, points, K, x_c1c2, x_bc)
    plt.show()
