import numpy as np
from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from matplotlib.patches import Polygon

poly0 = np.array([(1,1),(4,3),(5,1)])
poly1 = np.array([(1,4),(4,6),(5,4),(5,7),(1,7)])

def polygonRes(p, x):
    min_dist = 10000.
    min_res = None
    for i in range(p.shape[0]):
        j = (i+1)%p.shape[0]
        p1 = p[i]
        p2 = p[j]
        r = (p2 - p1).dot(x - p1)
        r /= np.linalg.norm(p2 - p1)**2
        if r < 0:
            dist = np.linalg.norm(x - p1)
            res = p1 - x
        elif r > 1:
            dist = np.linalg.norm(p2 - x)
            res = p2 - x
        else:
            dist = np.sqrt( np.linalg.norm(x - p1) **2 - (r * np.linalg.norm(p2-p1) ) ** 2)
            res = r*(p2-p1) - (x - p1) 
        if(min_dist > dist):
            min_dist = dist
            min_res = res
    epsilon = 0.00001
    vec = min_res/(np.linalg.norm(min_res)+epsilon)
    vec = -vec * 1./(min_dist+epsilon)
    return vec


polygonRes(np.array([[0,0],[1,1.],[0,2.],[-1.,1]]), np.array([0.1,2.]))
p1 = np.array([[0,0],[5,5],[4,6.],[-1.,1]])
p2 = np.array([[-5,-1],[5,-1],[5,-2],[-5.,-2]])

class pose2polygonEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(3)

    def residual(self, nodes):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (15)(16)
        """
        delta = 0.0001
        x = nodes[self.i].x[0:2]
        r = polygonRes(self.z, x)
        deltax0 = x + np.array([delta,0])
        deltax1 = x + np.array([0,delta])
        col0 = (polygonRes(self.z, deltax0) - r)/delta
        col1 = (polygonRes(self.z, deltax1) - r)/delta
        res = np.zeros(3)
        J = np.zeros([3,3])
        res[0:2] = r
        J[0:2,0] = col0
        J[0:2,1] = col1
        return res, J

class pose2dbetweenEdge:
    def __init__(self, i, j, z, omega = None, color = 'black'):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.color = color
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(self.z.shape[0])
    def residual(self, nodes):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (15)(16)
        """
        T12 = np.linalg.inv(v2m(nodes[self.i].x)).dot(v2m(nodes[self.j].x))
        T21 = np.linalg.inv(T12)
        R21,t21 = makeRt(T21)
        J = np.eye(3)
        J[0:2,0:2] = R21
        J[0:2,2] = -np.array([-t21[1], t21[0]])
        J = -J
        return m2v(np.linalg.inv(v2m(self.z)).dot(T12)), J, np.eye(3)


class pose2Node:
    def __init__(self, x):
        self.x = x
        self.size = x.size

    def update(self, dx):
        self.x = m2v(v2m(self.x).dot(v2m(dx)))

def draw(figname, graph):
    fig = plt.figure(figname)
    axes = fig.gca()
    for n in graph.nodes:
        plot_pose2(figname, v2m(n.x), 0.2)
    for e in graph.edges:
        if(e.type=='one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt(v2m(graph.nodes[i].x))
        _, tj = makeRt(v2m(graph.nodes[j].x))
        x = [ti[0],tj[0]]
        y = [ti[1],tj[1]]
        axes.plot(x,y,c=e.color,linestyle=':')


if __name__ == '__main__':
    
    graph = graphSolver()

    n = 10
    cur_pose = np.array([-5,-0.5,0])
    odom = np.array([1., 0, 0])
    for i in range(n):
        graph.addNode(pose2Node(cur_pose)) # add node to graph
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))

    for i in range(n-1):
        j = (i + 1)
        graph.addEdge(pose2dbetweenEdge(i,j,odom)) # add edge(i,j) to graph

    for i in range(n):
        graph.addEdge(pose2polygonEdge(i,p1)) #
        graph.addEdge(pose2polygonEdge(i,p2)) #



    draw('before loop-closing', graph)
    fig = plt.figure('before loop-closing')
    axes = fig.gca()
    polygon1 = Polygon(p1)
    polygon2 = Polygon(p2)
    
    axes.add_patch(polygon1)
    axes.add_patch(polygon2)
    graph.report()
    graph.solve()
    draw('after loop-closing', graph)
    fig = plt.figure('after loop-closing')
    axes = fig.gca()
    polygon1 = Polygon(p1)
    polygon2 = Polygon(p2)
    axes.add_patch(polygon1)
    axes.add_patch(polygon2)

    plt.show()
