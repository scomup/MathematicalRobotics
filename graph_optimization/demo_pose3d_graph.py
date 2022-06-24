import numpy as np
from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *

class pose3dEdge:
    def __init__(self, i, z, omega = None):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
        if(self.omega is None):
            self.omega = np.eye(self.z.shape[0])

    def residual(self, nodes):
        """
        The proof of Jocabian of SE3 is given in a graph_optimization.md (20)(21)
        """
        Tzx = np.linalg.inv(expSE3(self.z)).dot(expSE3(nodes[self.i].x))
        return logSE3(Tzx), np.eye(6)


class pose3dbetweenEdge:
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
        The proof of Jocabian of SE3 is given in a graph_optimization.md (20)(21)
        """
        T1 = expSE3(nodes[self.i].x)
        T2 = expSE3(nodes[self.j].x)

        T12 = np.linalg.inv(T1).dot(T2)
        T21 = np.linalg.inv(T12)
        R,t = makeRt(T21)
        J = np.zeros([6,6])
        J[0:3,0:3] = R
        J[0:3,3:6] = skew(t).dot(R)
        J[3:6,3:6] = R
        J = -J
        return logSE3(np.linalg.inv(expSE3(self.z)).dot(T12)), J, np.eye(6)

class pose3Node:
    def __init__(self, x):
        self.x = x
        self.size = x.size
        self.loc = 0

    def update(self, dx):
        self.x = logSE3(expSE3(self.x).dot(expSE3(dx)))

def draw(figname, gs):
    for n in gs.nodes:
        plot_pose3(figname, expSE3(n.x), 0.05)
    fig = plt.figure(figname)
    axes = fig.gca()
    for e in gs.edges:
        if(e.type=='one'):
            continue
        i = e.i
        j = e.j
        _, ti = makeRt(expSE3(gs.nodes[i].x))
        _, tj = makeRt(expSE3(gs.nodes[j].x))
        x = [ti[0],tj[0]]
        y = [ti[1],tj[1]]
        z = [ti[2],tj[2]]
        axes.plot(x,y,z,c=e.color,linestyle=':')
    set_axes_equal(figname)




if __name__ == '__main__':
    
    gs = graphSolver()

    n = 12
    cur_pose = np.array([0,0,0,0,0,0])
    odom = np.array([0.2, 0, 0.00, 0.05, 0, 0.5])
    for i in range(n):
        gs.addNode(pose3Node(cur_pose)) # add node to graph
        cur_pose = logSE3(expSE3(cur_pose).dot(expSE3(odom)))

    gs.addEdge(pose3dEdge(0,np.array([0,0,0,0,0,0]))) # add prior pose to graph

    for i in range(n-1):
        j = (i + 1)
        gs.addEdge(pose3dbetweenEdge(i,j,odom)) # add edge(i,j) to graph
        
    gs.addEdge(pose3dbetweenEdge(n-1, 0, odom, color='red'))


    draw('before loop-closing', gs)
    gs.solve()
    draw('after loop-closing', gs)

    plt.show()
