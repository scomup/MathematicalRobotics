import numpy as np

class graphSolver:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.ops = []
        self.loc = []
        self.psize = 0

    def addNode(self, node):
        self.nodes.append(node[0])
        self.ops.append(node[1])
        self.loc.append(self.psize)
        self.psize += node[0].size
        return len(self.nodes) - 1

    def addEdge(self, edge):
        self.edges.append(edge)

    def solve_once(self):
        H = np.zeros([self.psize,self.psize])
        g = np.zeros([self.psize])
        score = 0
        for edge in self.edges:
            if(len(edge) == 4):
                i, j, z, func = edge
                r, jacobian_i, jacobian_j = func(self.nodes[i], self.nodes[j], z)
                jacobian = np.zeros([3, self.psize])
                jacobian[:,self.loc[i]:self.loc[i]+3] = jacobian_i
                jacobian[:,self.loc[j]:self.loc[j]+3] = jacobian_j
                H += jacobian.T.dot(jacobian)
                g+= jacobian.T.dot(r)
                score += r.dot(r)
            elif(len(edge) == 3):
                i,  z, func = edge
                r, jacobian_i = func(self.nodes[i], z)
                jacobian = np.zeros([3, self.psize])
                jacobian[:,self.loc[i]:self.loc[i]+3] = jacobian_i
                H += jacobian.T.dot(jacobian)
                g+= jacobian.T.dot(r)
                score += r.dot(r)
        H_inv = np.linalg.inv(H)
        dx = np.dot(H_inv, -g)
        print(score)
        return dx

    def update(self, dx):
        for i, node in enumerate(self.nodes):
            self.nodes[i] = self.ops[i](node,dx[self.loc[i]:self.loc[i]+3])

if __name__ == '__main__':
    from math_tools import *

    def func1(x1, x2, z):
        T12 = np.linalg.inv(v2m(x1)).dot(v2m(x2))
        T21 = np.linalg.inv(T12)
        R21,t21 = makeRt(T21)
        Ad_T21 = np.eye(3)
        Ad_T21[0:2,0:2] = R21
        Ad_T21[0:2,2] = np.array([t21[1], -t21[0]])
        return m2v(np.linalg.inv(v2m(z)).dot(T12)), -Ad_T21, np.eye(3)

    def func2(x2, z):
        Tz2 = np.linalg.inv(v2m(z)).dot(v2m(x2))
        T2z = np.linalg.inv(Tz2)
        R,t = makeRt(T2z)
        Ad_T = np.eye(3)
        Ad_T[0:2,0:2] = R
        Ad_T[0:2,2] = np.array([t[1], -t[0]])
        return m2v(T2z), -Ad_T
    def plus(x1, x2):
        return m2v(v2m(x1).dot(v2m(x2)))

    gs = graphSolver()
    gs.addNode([np.array([0,0,0]),plus]) #0
    gs.addNode([np.array([1,0,0]),plus]) #1
    gs.addNode([np.array([0.5,0,0]),plus]) #2
    gs.addEdge([0,np.array([0,0,0]),func2]) #i, z, func
    gs.addEdge([1,np.array([1.1,0,0]),func2]) #i, z, func
    gs.addEdge([0,1,np.array([1.05,0,0]),func1]) #i, j, z, func
    gs.addEdge([1,2,np.array([-0.6,0,0]),func1]) #i, j, z, func
    gs.addEdge([0,2,np.array([0,0,0]),func1]) #i, j, z, func
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    gs.update(dx)
    dx = gs.solve_once()
    print(gs.nodes)
    


    

