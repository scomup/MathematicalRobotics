import numpy as np

class graphSolver:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.ops = []
        self.loc = []
        self.psize = 0

    def addNode(self, node):
        node.loc = self.psize
        self.psize += node.size
        self.nodes.append(node)
        return len(self.nodes) - 1

    def addEdge(self, edge):
        self.edges.append(edge)

    def solve_once(self):
        H = np.zeros([self.psize,self.psize])
        g = np.zeros([self.psize])
        score = 0
        for edge in self.edges:
            jacobian = np.zeros([3, self.psize])
            if(edge.type == 'between'):
                r, jacobian_i, jacobian_j = edge.func(self.nodes)
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                jacobian[:, node_i.loc : node_i.loc +  node_i.size] = jacobian_i
                jacobian[:, node_j.loc : node_j.loc +  node_j.size] = jacobian_j
            elif(edge.type == 'one'):
                node_i = self.nodes[edge.i]
                r, jacobian_i = edge.func(self.nodes)
                jacobian[:,node_i.loc:node_i.loc + node_i.size] = jacobian_i
            H += jacobian.T.dot(jacobian)
            g+= jacobian.T.dot(r)
            score += r.dot(r)
        H_inv = np.linalg.inv(H)
        dx = np.dot(H_inv, -g)
        print(score)
        return dx

    def update(self, dx):
        for i, node in enumerate(self.nodes):
            node.update(dx[node.loc:node.loc+node.size])


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
    from math_tools import *

    gs = graphSolver()
    gs.addNode(pose2Node(np.array([0,0,0]))) #0
    gs.addNode(pose2Node(np.array([1,0,0]))) #1
    gs.addNode(pose2Node(np.array([0.5,0,0]))) #2
    gs.addEdge(pose2dEdge(0,np.array([0,0,0]))) #i, z, func
    gs.addEdge(pose2dEdge(1,np.array([1.1,0,0]))) #i, z, func
    gs.addEdge(pose2dbetweenEdge(0,1,np.array([1.05,0,0]))) #i, j, z, func
    gs.addEdge(pose2dbetweenEdge(1,2,np.array([-0.6,0,0]))) #i, j, z, func
    gs.addEdge(pose2dbetweenEdge(0,2,np.array([0,0,0]))) #i, j, z, func
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    print(gs.nodes)
    


    

