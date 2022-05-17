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
            #jacobian = np.zeros([3, self.psize])
            if(edge.type == 'two'):
                r, jacobian_i, jacobian_j = edge.func(self.nodes)
                node_i = self.nodes[edge.i]
                node_j = self.nodes[edge.j]
                H[node_i.loc:node_i.loc+node_i.size,node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(jacobian_i) 
                H[node_j.loc:node_j.loc+node_j.size,node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(jacobian_j) 
                H[node_i.loc:node_i.loc+node_i.size,node_j.loc:node_j.loc+node_j.size] += jacobian_i.T.dot(jacobian_j)  
                H[node_j.loc:node_j.loc+node_j.size,node_i.loc:node_i.loc+node_i.size] += jacobian_j.T.dot(jacobian_i)  
                g[node_i.loc:node_i.loc+node_i.size] += jacobian_i.T.dot(r)
                g[node_j.loc:node_j.loc+node_j.size] += jacobian_j.T.dot(r)
                score += r.dot(r)

            elif(edge.type == 'one'):
                node_i = self.nodes[edge.i]
                r, jacobian_i = edge.func(self.nodes)
                H[node_i.loc:node_i.loc+ node_i.size,node_i.loc:node_i.loc+ node_i.size] += jacobian_i.T.dot(jacobian_i) 
                g[node_i.loc:node_i.loc+ node_i.size] += jacobian_i.T.dot(r)
                score += r.dot(r)
        dx = np.linalg.solve(H, -g)
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

    


    

