import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.robust_kernel import *


class GraphSolver:
    """
    A graph optimization solver.
    more information is written in graph_optimization.md
    """
    def __init__(self, use_sparse = False):
        self.vertices = []
        self.is_no_constant = []
        self.edges = []
        self.loc = []
        self.psize = 0
        self.use_sparse = use_sparse

    def add_vertex(self, vertex, is_constant = False):
        self.vertices.append(vertex)
        if (not is_constant):
            self.loc.append(self.psize)
            self.psize += vertex.size
        else:
            self.loc.append(np.nan)
        self.is_no_constant.append(not is_constant)
        return len(self.vertices) - 1
    
    def set_constant(self, idx):
        self.psize -= self.vertices[idx].size
        self.loc[idx] = np.nan
        self.is_no_constant[idx] = False
        for i in range(idx, len(self.loc)):
            self.loc[i] -= self.vertices[idx].size

    def add_edge(self, edge):
        self.edges.append(edge)

    def report(self):
        error = 0
        type_score = {}
        for edge in self.edges:
            r = edge.residual(self.vertices)[0]
            omega = edge.omega
            try:
                kernel = edge.kernel
                if (kernel is None):
                    kernel = L2Kernel()
            except:
                kernel = L2Kernel()
            e2 = r @ omega @ r
            rho = kernel.apply(e2)
            error += rho[0]
            edge_type_name = type(edge).__name__
            if (edge_type_name in type_score):
                type_score[edge_type_name] += rho[0]
            else:
                type_score.setdefault(edge_type_name, rho[0])

        print("---------------------")
        print("The number of parameters: %d." % self.psize)
        print("The number of vertices: %d." % len(self.vertices))
        print("The number of edges: %d." % len(self.edges))
        print("Overall error: %f." % error)
        type_list = list(type_score)
        for t in type_list:
            # print("  -> %20s: %5f." % (t, type_score[t]))
            print(' -> {:<20}: {:<.4f}'.format(t, type_score[t]))
        print("---------------------")

    def solve_once(self):
        H = np.zeros([self.psize, self.psize])
        g = np.zeros([self.psize])
        score = 0
        for edge in self.edges:
            # self.vertices[edge.i]
            omega = edge.omega
            try:
                kernel = edge.kernel
                if (kernel is None):
                    kernel = L2Kernel()
            except:
                kernel = L2Kernel()
            if (edge.type == 'one'):
                vertex_i = self.vertices[edge.i]
                r, jacobian_i = edge.residual(self.vertices)
                e2 = r @ omega @ r
                rho = kernel.apply(e2)
                s_i = self.loc[edge.i]
                e_i = s_i + vertex_i.size
                if (self.is_no_constant[edge.i]):
                    H[s_i:e_i, s_i:e_i] += rho[1] * jacobian_i.T @ omega @ jacobian_i
                    g[s_i:e_i] += rho[1] * jacobian_i.T @ omega @ r
            elif (edge.type == 'two'):
                r, jacobian_i, jacobian_j = edge.residual(self.vertices)
                e2 = r @ omega @ r
                rho = kernel.apply(e2)
                vertex_i = self.vertices[edge.i]
                vertex_j = self.vertices[edge.j]
                s_i = self.loc[edge.i]
                s_j = self.loc[edge.j]
                e_i = s_i + vertex_i.size
                e_j = s_j + vertex_j.size
                if (self.is_no_constant[edge.i]):
                    H[s_i:e_i, s_i:e_i] += rho[1]*jacobian_i.T @ omega @ jacobian_i
                    g[s_i:e_i] += rho[1]*jacobian_i.T @ omega @ r
                if (self.is_no_constant[edge.j]):
                    H[s_j:e_j, s_j:e_j] += rho[1]*jacobian_j.T @ omega @ jacobian_j
                    g[s_j:e_j] += rho[1]*jacobian_j.T @ omega @ r
                if (self.is_no_constant[edge.j] and self.is_no_constant[edge.i]): 
                    H[s_i:e_i, s_j:e_j] += rho[1]*jacobian_i.T @ omega @ jacobian_j
                    H[s_j:e_j, s_i:e_i] += rho[1]*jacobian_j.T @ omega @ jacobian_i
            elif (edge.type == 'three'):
                vertex_i = self.vertices[edge.i]
                vertex_j = self.vertices[edge.j]
                vertex_k = self.vertices[edge.k]
                s_i = self.loc[edge.i]
                s_j = self.loc[edge.j]
                s_k = self.loc[edge.k]
                e_i = s_i + vertex_i.size
                e_j = s_j + vertex_j.size
                e_k = s_k + vertex_k.size
                r, jacobian_i, jacobian_j, jacobian_k = edge.residual(self.vertices)
                e2 = r @ omega @ r
                rho = kernel.apply(e2)
                if (self.is_no_constant[edge.i]):
                    H[s_i:e_i, s_i:e_i] += rho[1]*jacobian_i.T @ omega @ jacobian_i
                    g[s_i:e_i] += rho[1]*jacobian_i.T @ omega @ r
                if (self.is_no_constant[edge.j]):
                    H[s_j:e_j, s_j:e_j] += rho[1]*jacobian_j.T @ omega @ jacobian_j
                    g[s_j:e_j] += rho[1]*jacobian_j.T @ omega @ r
                if (self.is_no_constant[edge.k]):
                    H[s_k:e_k, s_k:e_k] += rho[1]*jacobian_k.T @ omega @ jacobian_k
                    g[s_k:e_k] += rho[1]*jacobian_k.T @ omega @ r
                if (self.is_no_constant[edge.i] and self.is_no_constant[edge.j]):
                    H[s_i:e_i, s_j:e_j] += rho[1]*jacobian_i.T @ omega @ jacobian_j
                    H[s_j:e_j, s_i:e_i] += rho[1]*jacobian_j.T @ omega @ jacobian_i
                if (self.is_no_constant[edge.i] and self.is_no_constant[edge.k]):
                    H[s_i:e_i, s_k:e_k] += rho[1]*jacobian_i.T @ omega @ jacobian_k
                    H[s_k:e_k, s_i:e_i] += rho[1]*jacobian_k.T @ omega @ jacobian_i
                if (self.is_no_constant[edge.j] and self.is_no_constant[edge.k]):
                    H[s_j:e_j, s_k:e_k] += rho[1]*jacobian_j.T @ omega @ jacobian_k
                    H[s_k:e_k, s_j:e_j] += rho[1]*jacobian_k.T @ omega @ jacobian_j
            score += rho[0]
        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(H), vmax=np.average(np.abs(H)[np.nonzero(np.abs(H))]))
        # plt.imshow(np.linalg.inv(H))
        # plt.plot(g)
        # plt.show()
        # dx = np.linalg.solve(H, -g)
        # much faster than np.linalg.solve!
        if (self.use_sparse):
            dx = spsolve(csc_matrix(H, dtype=float), csc_matrix(-g, dtype=float).T)
        else:
            try:
                dx = np.linalg.solve(H, -g)
            except:
                # print('Bad Hassian matrix!')
                dx = np.linalg.pinv(H) @ -g
        return dx, score

    def solve(self, show_info=True, min_score_change=0.01, step=0):
        last_score = np.inf
        iter = 0
        while(True):
            dx, score = self.solve_once()
            # import matplotlib.pyplot as plt
            # plt.plot(dx)
            # plt.show()

            if (step > 0 and np.linalg.norm(dx) > step):
                dx = dx/np.linalg.norm(dx)*step
            iter += 1
            if (show_info):
                print('iter %d: %f' % (iter, score))
            if (last_score - score < min_score_change):
                break
            self.update(dx)
            last_score = score

    def update(self, dx):
        for i, vertex in enumerate(self.vertices):
            if self.is_no_constant[i]:
                s_i = self.loc[i]
                e_i = s_i + vertex.size
                vertex.update(dx[s_i:e_i])
