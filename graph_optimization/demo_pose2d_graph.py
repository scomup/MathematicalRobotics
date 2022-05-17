import numpy as np
from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from plot_pose2d import *

class pose2dEdge:
    def __init__(self, i, z):
        self.i = i
        self.z = z
        self.type = 'one'
    def func(self, nodes):
        Tzx = np.linalg.inv(v2m(self.z)).dot(v2m(nodes[self.i].x))
        return m2v(Tzx), np.eye(3)


class pose2dbetweenEdge:
    def __init__(self, i, j, z):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'two'
    def func(self, nodes):
        T12 = np.linalg.inv(v2m(nodes[self.i].x)).dot(v2m(nodes[self.j].x))
        T21 = np.linalg.inv(T12)
        R21,t21 = makeRt(T21)
        Ad_T21 = np.eye(3)
        Ad_T21[0:2,0:2] = R21
        Ad_T21[0:2,2] = np.array([t21[1], -t21[0]])
        return m2v(np.linalg.inv(v2m(self.z)).dot(T12)), -Ad_T21, np.eye(3)

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
    for i in range(n):
        j = (i + 1)%n
        gs.addEdge(pose2dbetweenEdge(i,j,odom))
        plot_pose2(figname, gs.nodes[i].x, 0.05)
        x = [gs.nodes[i].x[0],gs.nodes[j].x[0]]
        y = [gs.nodes[i].x[1],gs.nodes[j].x[1]]
        if(j!=0):
            axes.plot(x,y,c='black',linestyle=':')
        else:
            axes.plot(x,y,c='r',linestyle=':')


if __name__ == '__main__':
    
    gs = graphSolver()

    n = 12
    cur_pose = np.array([0,0,0])
    odom = np.array([0.2, 0, 0.45])
    for i in range(n):
        gs.addNode(pose2Node(cur_pose)) # add node to graph
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))

    gs.addEdge(pose2dEdge(0,np.array([0,0,0]))) # add prior pose to graph

    for i in range(n):
        j = (i + 1)%n
        gs.addEdge(pose2dbetweenEdge(i,j,odom)) # add edge(i,j) to graph


    draw('before loop-closing', gs)
    gs.solve()
    draw('after loop-closing', gs)

    plt.show()
