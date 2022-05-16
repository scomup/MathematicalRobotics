import numpy as np
from graph_solver import *
from math_tools import *
import gtsam.utils.plot as gtsam_plot
import gtsam

class pose2dEdge:
    def __init__(self, i, z):
        self.i = i
        self.z = z
        self.type = 'one'
    def func(self, nodes):
        Tz2 = np.linalg.inv(v2m(self.z)).dot(v2m(nodes[self.i].x))
        T2z = np.linalg.inv(Tz2)
        R,t = makeRt(T2z)
        Ad_T = np.eye(3)
        Ad_T[0:2,0:2] = R
        Ad_T[0:2,2] = np.array([t[1], -t[0]])
        return m2v(T2z), -Ad_T


class pose2dbetweenEdge:
    def __init__(self, i, j, z):
        self.i = i
        self.j = j
        self.z = z
        self.type = 'between'
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



if __name__ == '__main__':
    gs = graphSolver()
    gs.addNode(pose2Node(np.array([0,0,0]))) #0
    gs.addNode(pose2Node(np.array([1,0,np.pi/2]))) #1
    gs.addNode(pose2Node(np.array([1,1,np.pi]))) #2
    gs.addNode(pose2Node(np.array([0,1,-np.pi/2]))) #3
    gs.addEdge(pose2dEdge(0,np.array([0,0,0]))) #i, z
    #gs.addEdge(pose2dEdge(1,np.array([1.1,0,0]))) #i, z
    #gs.addEdge(pose2dEdge(2,np.array([0.6,0,0]))) #i, z
    gs.addEdge(pose2dbetweenEdge(0,1,np.array([1,0,np.pi/2]))) #i, j, z
    gs.addEdge(pose2dbetweenEdge(1,2,np.array([1,0,np.pi/2]))) #i, j, z
    gs.addEdge(pose2dbetweenEdge(2,3,np.array([1,0,np.pi/2]))) #i, j, z
    gs.addEdge(pose2dbetweenEdge(3,0,np.array([0,0,np.pi/2]))) #i, j, z

    
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    for n in gs.nodes:
        print(n.x)
    import matplotlib.pyplot as plt
    for n in gs.nodes:
        gtsam_plot.plot_pose2(0, gtsam.Pose2(*n.x), 0.1)
    plt.axis('equal')
    plt.show()
