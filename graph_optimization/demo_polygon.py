import numpy as np
from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from utilities.polygon import *
from matplotlib.patches import Polygon
from graph_optimization.demo_pose2d_graph import pose2dEdge, pose2dbetweenEdge


class pose2polygonEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(3)

    def residual(self, nodes):
        x = nodes[self.i].x[0:2]
        r = polygonRes(x,self.z)
        res = np.zeros(3)
        J = np.zeros([3,3])
        J[0:2,0:2] = numericalDerivative(polygonRes, x, self.z)
        res[0:2] = r
        return res, J



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

def draw_polygon(figname, ploygons):    
    fig = plt.figure(figname)
    axes = fig.gca()
    for poly in ploygons:
        polygon = Polygon(poly)
        axes.add_patch(polygon)

if __name__ == '__main__':
    ploygons = []

    ploygons.append(np.array([[-5,3],[5,3],[5,1],[-5.,1]]))
    ploygons.append(np.array([[-5,-1],[5,-1],[5,-3],[-5.,-3]]))
    ploygons.append(np.array([[7,-3],[10,-3],[10,5]]))
    
    graph = graphSolver()
    n = 20
    
    cur_pose = np.array([-6,-0.5,0])
    odom = np.array([0.8, 0, 0.025])
    for i in range(n):
        graph.addNode(pose2Node(cur_pose)) # add node to graph
        if(i == 0):
            graph.addEdge(pose2dEdge(i,cur_pose))    
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))

    for i in range(n-1):
        j = (i + 1)
        graph.addEdge(pose2dbetweenEdge(i,j,odom)) # add edge(i,j) to graph

    for i in range(n):
        for poly in ploygons:
            graph.addEdge(pose2polygonEdge(i,poly)) #



    draw('before loop-closing', graph)
    draw_polygon('before loop-closing', ploygons)

    graph.report()
    graph.solve()
    draw('after loop-closing', graph)
    draw_polygon('after loop-closing', ploygons)

    plt.show()
