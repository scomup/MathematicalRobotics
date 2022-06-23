import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

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
        node.loc = self.psize
        self.psize += node.size
        self.nodes.append(node)
        return len(self.nodes) - 1

    def addEdge(self, edge):
        self.edges.append(edge)

    def getScore(self):
        score = 0
        for edge in self.edges:
            if(edge.type == 'one'):
                r, _ = edge.residual(self.nodes)
                score += r.dot(r)
            elif(edge.type == 'two'):
                r, _, _ = edge.residual(self.nodes)
                score += r.dot(r)
            elif(edge.type == 'three'):
                r, _, _, _ = edge.residual(self.nodes)
                score += r.dot(r)
        return score


    def solve_once(self):
        H = np.zeros([self.psize,self.psize])
        g = np.zeros([self.psize])
        score = 0
        for edge in self.edges:
            if(edge.type == 'one'):
                node_i = self.nodes[edge.i]
                r, jacobian_i = edge.residual(self.nodes)
                H[node_i.loc:node_i.loc+ node_i.size,node_i.loc:node_i.loc+ node_i.size] += jacobian_i.T.dot(jacobian_i) 
                g[node_i.loc:node_i.loc+ node_i.size] += jacobian_i.T.dot(r)
                score += r.dot(r)
            elif(edge.type == 'two'):
                r, jacobian_i, jacobian_j = edge.residual(self.nodes)
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                H[node_i.loc:node_i.loc+node_i.size,node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(jacobian_i) 
                H[node_j.loc:node_j.loc+node_j.size,node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(jacobian_j) 
                H[node_i.loc:node_i.loc+node_i.size,node_j.loc:node_j.loc+node_j.size] += jacobian_i.T.dot(jacobian_j)  
                H[node_j.loc:node_j.loc+node_j.size,node_i.loc:node_i.loc+node_i.size] += jacobian_j.T.dot(jacobian_i)  
                g[node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(r)
                g[node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(r)
                score += r.dot(r)
            elif(edge.type == 'three'):
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                node_k = self.nodes[edge.k]
                r, jacobian_i, jacobian_j, jacobian_k = edge.residual(self.nodes)
                H[node_i.loc:node_i.loc+node_i.size,node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(jacobian_i) 
                H[node_j.loc:node_j.loc+node_j.size,node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(jacobian_j) 
                H[node_k.loc:node_k.loc+node_k.size,node_k.loc:node_k.loc+node_k.size] += jacobian_k.T.dot(jacobian_k) 
                H[node_i.loc:node_i.loc+node_i.size,node_j.loc:node_j.loc+node_j.size] += jacobian_i.T.dot(jacobian_j)  
                H[node_j.loc:node_j.loc+node_j.size,node_i.loc:node_i.loc+node_i.size] += jacobian_j.T.dot(jacobian_i)  
                H[node_i.loc:node_i.loc+node_i.size,node_k.loc:node_k.loc+node_k.size] += jacobian_i.T.dot(jacobian_k)  
                H[node_k.loc:node_k.loc+node_k.size,node_i.loc:node_i.loc+node_i.size] += jacobian_k.T.dot(jacobian_i)  
                H[node_j.loc:node_j.loc+node_j.size,node_k.loc:node_k.loc+node_k.size] += jacobian_j.T.dot(jacobian_k)  
                H[node_k.loc:node_k.loc+node_k.size,node_j.loc:node_j.loc+node_j.size] += jacobian_k.T.dot(jacobian_j)  
                g[node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(r)
                g[node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(r)
                g[node_k.loc:node_k.loc+node_k.size] += jacobian_k.T.dot(r)
                score += r.dot(r)
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
            node.update(dx[node.loc:node.loc+node.size])

    


    

