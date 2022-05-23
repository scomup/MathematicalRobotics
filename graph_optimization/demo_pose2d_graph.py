import numpy as np
from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *

class pose2dEdge:
    def __init__(self, i, z):
        self.i = i
        self.z = z
        self.type = 'one'
    def func(self, nodes):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (15)(16)
        """
        Tzx = np.linalg.inv(v2m(self.z)).dot(v2m(nodes[self.i].x))
        return m2v(Tzx), np.eye(3)


class pose2dbetweenEdge:
    def __init__(self, i, j, z, color = 'black'):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
        self.color = color
    def func(self, nodes):
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
        self.loc = 0

    def update(self, dx):
        self.x = m2v(v2m(self.x).dot(v2m(dx)))

def draw(figname, gs):
    fig = plt.figure(figname)
    axes = fig.gca()
    for n in gs.nodes:
        plot_pose2(figname, n.x, 0.05)
    for e in gs.edges:
        if(e.type=='one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt(v2m(gs.nodes[i].x))
        _, tj = makeRt(v2m(gs.nodes[j].x))
        x = [ti[0],tj[0]]
        y = [ti[1],tj[1]]
        axes.plot(x,y,c=e.color,linestyle=':')


if __name__ == '__main__':
    
    gs = graphSolver()

    n = 12
    cur_pose = np.array([0,0,0])
    odom = np.array([0.2, 0, 0.45])
    for i in range(n):
        gs.addNode(pose2Node(cur_pose)) # add node to graph
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))

    gs.addEdge(pose2dEdge(0,np.array([0,0,0]))) # add prior pose to graph

    for i in range(n-1):
        j = (i + 1)
        gs.addEdge(pose2dbetweenEdge(i,j,odom)) # add edge(i,j) to graph
        
    gs.addEdge(pose2dbetweenEdge(n-1, 0, odom, 'red'))


    draw('before loop-closing', gs)
    gs.solve()
    draw('after loop-closing', gs)

    plt.show()
