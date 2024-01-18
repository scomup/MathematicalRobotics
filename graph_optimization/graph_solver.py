import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.linalg import cho_solve, cho_factor
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, eye
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


class Hassian:
    """
    A more memory-efficient Hessian for big graph.
    """
    def __init__(self, graph):
        self.use_sparse = graph.use_sparse
        self.psize = graph.psize
        self.loc = graph.loc
        self.epsilon = graph.epsilon
        if (not self.use_sparse):
            self.H = np.zeros([self.psize, self.psize])
        else:
            self.H_dict = {}
            self.m = len(graph.vertices)

    def add(self, i, j, s_i, e_i, s_j, e_j, h):
        """
        Because the Hessian matrix is symmetric,
        we only need to store the upper triangular part.
        """
        if i >= j:
            if (not self.use_sparse):
                self.H[s_i:e_i, s_j:e_j] += h
                if (i != j):
                    self.H[s_j:e_j, s_i:e_i] += h.T
            else:
                key = int(i * self.m + j)
                if key in self.H_dict:
                    self.H_dict[key] += h
                else:
                    self.H_dict.update({key: h})

    def matrix(self):
        if (not self.use_sparse):
            self.H.flat[::self.H.shape[0]+1] += self.epsilon
            return self.H
        else:
            rows = []
            cols = []
            values = []
            for key, h in self.H_dict.items():
                v_i = int(key / self.m)
                v_j = int(key % self.m)
                s_i = self.loc[v_i]
                s_j = self.loc[v_j]
                row, col = np.nonzero(h)
                val = h[row, col].tolist()
                row = (row + s_i).tolist()
                col = (col + s_j).tolist()
                cols += col
                rows += row
                values += val
                if (v_i != v_j):
                    # Convert to symmetric matrix
                    cols += row
                    rows += col
                    values += val
            H = coo_matrix((values, (rows, cols)), shape=(self.psize, self.psize))
            H_regularized = csc_matrix(H) + eye(self.psize) * self.epsilon
            return H_regularized


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
        start = time.time()
        g = np.zeros([self.psize])
        H = Hassian(self)

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
                    s_j = self.loc[v_j]
                    e_j = self.loc[v_j] + self.vertices[v_j].size
                    jacobian_j = jacobian[j]
                    if (self.is_no_constant[v_j] and self.is_no_constant[v_i]):
                        if v_i >= v_j:
                            h = rho[1] * jacobian_i.T @ omega @ jacobian_j
                            H.add(v_i, v_j, s_i, e_i, s_j, e_j, h)
            score += rho[0]

        H = H.matrix()

        if (self.use_sparse):
            dx = cholesky(H).solve_A(-g)
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
