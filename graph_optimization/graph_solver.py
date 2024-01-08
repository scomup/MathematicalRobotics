import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sksparse.cholmod import cholesky
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.robust_kernel import *
import time

class BaseVertex:
    def __init__(self, x, size):
        self.x = x  # vertex object
        self.size = size  # vertex object's size

    def update(self, dx):
        '''
        implemented the update function for vertex
        '''
        print("not implemented!")
        self.x = self.x + dx


class BaseEdge:
    def __init__(self, link, z, omega, kernel):
        self.link = link
        self.z = z
        self.omega = omega
        self.kernel = kernel

    def residual(self, vertices):
        print("not implemented!")
        '''
        calculate the residual and jacbians for edge
        '''
        r = 0.  # residual
        J = []  # jacbians
        for v_i in self.link:
            J.append(0)  # dr_dxi
        return r, J


class GraphSolver:
    """
    A graph optimization solver.
    more information is written in graph_optimization.md
    """
    def __init__(self, use_sparse=False, epsilon=1e-6):
        self.vertices = []
        self.is_no_constant = []
        self.edges = []
        self.loc = []
        self.psize = 0
        self.use_sparse = use_sparse
        self.epsilon = epsilon

    def add_vertex(self, vertex, is_constant=False):
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
            if (hasattr(edge, 'kernel') and edge.kernel is not None):
                kernel = edge.kernel
            else:
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
        # H = lil_matrix((self.psize, self.psize))

        score = 0
        for edge in self.edges:
            # self.vertices[v_i]
            omega = edge.omega
            kernel = None
            if (hasattr(edge, 'kernel') and edge.kernel is not None):
                kernel = edge.kernel
            else:
                kernel = L2Kernel()

            link = edge.link
            r, jacobian = edge.residual(self.vertices)
            e2 = r @ omega @ r
            rho = kernel.apply(e2)

            for i, v_i in enumerate(edge.link):
                s_i = self.loc[v_i]
                e_i = self.loc[v_i] + self.vertices[v_i].size
                jacobian_i = jacobian[i]
                if (self.is_no_constant[v_i]):
                    g[s_i:e_i] += rho[1] * jacobian_i.T @ omega @ r
                for j, v_j in enumerate(edge.link):
                    s_i = self.loc[v_i]
                    e_i = self.loc[v_i] + self.vertices[v_i].size
                    s_j = self.loc[v_j]
                    e_j = self.loc[v_j] + self.vertices[v_j].size
                    jacobian_j = jacobian[j]
                    if (self.is_no_constant[v_j] and self.is_no_constant[v_i]):
                        H[s_i:e_i, s_j:e_j] += rho[1] * jacobian_i.T @ omega @ jacobian_j
            score += rho[0]
        # import matplotlib.pyplot as plt
        # plt.imshow(np.abs(H), vmax=np.average(np.abs(H)[np.nonzero(np.abs(H))]))
        # plt.imshow(np.linalg.inv(H))
        # plt.plot(g)
        # plt.show()

        H.flat[::H.shape[0]+1] += self.epsilon  # Regularization
        if (self.use_sparse):
            dx = cholesky(csc_matrix(H)).solve_A(-g)
        else:
            dx = np.linalg.solve(H, -g)
        return dx, score

    def solve(self, show_info=True, min_score_change=0.01, step=0):
        last_score = np.inf
        itr = 0
        while(True):
            start = time.time()
            dx, score = self.solve_once()
            end = time.time()
            if (step > 0 and np.max(dx) > step):
                dx = dx/np.max(dx) * step
            itr += 1
            if (show_info):
                time_diff = end - start
                print('iter %d: solve time: %f error: %f' % (itr, time_diff, score))
            if (last_score - score < min_score_change and itr > 5):
                break
            self.update(dx)
            last_score = score

    def update(self, dx):
        for i, vertex in enumerate(self.vertices):
            if self.is_no_constant[i]:
                s_i = self.loc[i]
                e_i = s_i + vertex.size
                vertex.update(dx[s_i:e_i])
