import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
#import time

class graphSolver:
    """
    A graph optimization solver.
    more information is written in graph_optimization.md
    """
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.ops = []
        self.loc = []
        self.psize = 0

    def addNode(self, node):
        self.loc.append(self.psize)
        self.nodes.append(node)
        self.psize += node.size
        return len(self.nodes) - 1

    def addEdge(self, edge):
        self.edges.append(edge)

    def getScore(self):
        score = 0
        for edge in self.edges:
            r = edge.residual(self.nodes)[0]
            omega = edge.omega
            score += r.dot(omega.dot(r))
        return score


    def solve_once(self):
        H = np.zeros([self.psize,self.psize])
        g = np.zeros([self.psize])
        score = 0
        for edge in self.edges:
            if(edge.type == 'one'):
                node_i = self.nodes[edge.i]
                r, jacobian_i = edge.residual(self.nodes)
                omega = edge.omega
                s_i = self.loc[edge.i]
                e_i = s_i + node_i.size
                H[s_i:e_i,s_i:e_i] += jacobian_i.T.dot(omega.dot(jacobian_i)) 
                g[s_i:e_i] += jacobian_i.T.dot(omega.dot(r))
                score += r.dot(omega.dot(r))
            elif(edge.type == 'two'):
                r, jacobian_i, jacobian_j = edge.residual(self.nodes)
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                omega = edge.omega
                s_i = self.loc[edge.i]
                s_j = self.loc[edge.j]
                e_i = s_i + node_i.size
                e_j = s_j + node_j.size
                H[s_i:e_i,s_i:e_i] += jacobian_i.T.dot(omega.dot(jacobian_i)) 
                H[s_j:e_j,s_j:e_j] += jacobian_j.T.dot(omega.dot(jacobian_j)) 
                H[s_i:e_i,s_j:e_j] += jacobian_i.T.dot(omega.dot(jacobian_j))  
                H[s_j:e_j,s_i:e_i] += jacobian_j.T.dot(omega.dot(jacobian_i))  
                g[s_i:e_i] += jacobian_i.T.dot(omega.dot(r))
                g[s_j:e_j] += jacobian_j.T.dot(omega.dot(r))
                score += r.dot(omega.dot(r))
            elif(edge.type == 'three'):
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                node_k = self.nodes[edge.k]
                omega = edge.omega
                s_i = self.loc[edge.i]
                s_j = self.loc[edge.j]
                s_k = self.loc[edge.k]
                e_i = s_i + node_i.size
                e_j = s_j + node_j.size
                e_k = s_k + node_k.size
                r, jacobian_i, jacobian_j, jacobian_k = edge.residual(self.nodes)
                H[s_i:e_i,s_i:e_i] += jacobian_i.T.dot(omega.dot(jacobian_i)) 
                H[s_j:e_j,s_j:e_j] += jacobian_j.T.dot(omega.dot(jacobian_j)) 
                H[s_k:e_k,s_k:e_k] += jacobian_k.T.dot(omega.dot(jacobian_k)) 
                H[s_i:e_i,s_j:e_j] += jacobian_i.T.dot(omega.dot(jacobian_j))  
                H[s_j:e_j,s_i:e_i] += jacobian_j.T.dot(omega.dot(jacobian_i))  
                H[s_i:e_i,s_k:e_k] += jacobian_i.T.dot(omega.dot(jacobian_k))  
                H[s_k:e_k,s_i:e_i] += jacobian_k.T.dot(omega.dot(jacobian_i))  
                H[s_j:e_j,s_k:e_k] += jacobian_j.T.dot(omega.dot(jacobian_k))  
                H[s_k:e_k,s_j:e_j] += jacobian_k.T.dot(omega.dot(jacobian_j))  
                g[s_i:e_i] += jacobian_i.T.dot(omega.dot(r))
                g[s_j:e_j] += jacobian_j.T.dot(omega.dot(r))
                g[s_k:e_k] += jacobian_k.T.dot(omega.dot(r))
                score += r.dot(omega.dot(r))
        #import matplotlib.pyplot as plt
        #plt.imshow(np.abs(H), vmax=np.average(np.abs(H)[np.nonzero(np.abs(H))]))
        #plt.show()
        #dx = np.linalg.solve(H, -g)
        #much faster than np.linalg.solve!
        dx = spsolve(csc_matrix(H, dtype=float), csc_matrix(-g, dtype=float).T)
        return dx, score

    def solve(self):
        last_score = None
        iter = 0
        while(True):   
            dx, score = self.solve_once()
            iter +=1
            print('iter %d: %f'%(iter, score))
            self.update(dx)
            if(last_score is None):
                last_score = score
                continue        
            if(last_score < score):
                break
            if(last_score - score < 0.0001):
                break
            last_score = score


    def update(self, dx):
        for i, node in enumerate(self.nodes):
            s_i = self.loc[i]
            e_i = s_i + node.size
            node.update(dx[s_i:e_i])

    


    

