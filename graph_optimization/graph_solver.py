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
            g += jacobian.T.dot(r)
            score += r.dot(r)
        #H_inv = np.linalg.inv(H)
        #dx = np.dot(H_inv, -g)
        dx = np.linalg.solve(H, -g)
        print(score)
        return dx

    def update(self, dx):
        for i, node in enumerate(self.nodes):
            node.update(dx[node.loc:node.loc+node.size])

    


    

