import numpy as np
from graph_solver import *
from math_tools import *
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
    gs.addEdge(pose2dEdge(0,np.array([0,0,0]))) 
    #gs.addEdge(pose2dEdge(1,np.array([1.1,0,0]))) #i, z
    #gs.addEdge(pose2dEdge(2,np.array([0.6,0,0]))) #i, z
    gs.addEdge(pose2dbetweenEdge(0,1,np.array([1,0,np.pi/2])))
    gs.addEdge(pose2dbetweenEdge(1,2,np.array([1,0,np.pi/2])))
    gs.addEdge(pose2dbetweenEdge(2,3,np.array([1,0,np.pi/2])))
    gs.addEdge(pose2dbetweenEdge(3,0,np.array([0,0,np.pi/2])))
    gs.solve()

    import matplotlib.pyplot as plt
    for n in gs.nodes:
        plot_pose2(0, n.x, 0.05)
    plt.axis('equal')
    plt.show()
