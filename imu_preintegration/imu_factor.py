from preintegration import *


class naviNode:
    def __init__(self, state):
        self.state = state
        self.size = 9
    def update(self, dx):
        d_state = navState(expSO3(dx[0:3]), dx[3:6], dx[6:9])
        self.state = self.state.retract(d_state)

class biasNode:
    def __init__(self, bias):
        self.bias = bias
        self.size = 6
    def update(self, dx):
        self.bias = self.bias + dx

class biasEdge:
    def __init__(self, i, z, omega = np.eye(6)):
        self.i = i #bias i
        self.type = 'one'
        self.z = z
        self.omega = omega
    def residual(self, nodes):
        bias_i = nodes[self.i].bias
        r = bias_i - self.z 
        return r, np.eye(6)

class biasbetweenEdge:
    def __init__(self, i, j, omega = np.eye(6)):
        self.i = i #bias i
        self.j = j #bias j
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        bias_i = nodes[self.i].bias
        bias_j = nodes[self.j].bias
        r = bias_i - bias_j 
        return r, np.eye(6), -np.eye(6)

class naviEdge:
    def __init__(self, i, z, omega = np.eye(9)):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
    def residual(self, nodes):
        state = nodes[self.i].state
        r, j, _ = state.local(self.z, True)
        r = r.vec()
        return r, j

class navibetweenEdge:
    def __init__(self, i, j, z, omega = np.eye(9)):
        self.i = i #state i
        self.j = j #state j
        self.z = z #error between state i and j
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        state_i = nodes[self.i].state
        state_j = nodes[self.j].state
        r, j1, j2 = state_i.local(state_j, True)
        r = self.z.local(r, False)
        r = r.vec()
        return r, j1, j2

class imuEdge:
    def __init__(self, i, j, k, z, omega = np.eye(9)):
        self.i = i #state i
        self.j = j #state j
        self.k = k #bias i
        self.z = z #pim between ij
        self.type = 'three'
        self.omega = omega
    def residual(self, nodes):
        pim = self.z
        statei = nodes[self.i].state
        statej = nodes[self.j].state
        bias = nodes[self.k].bias
        statejstar, J_statejstar_statei, J_statejstar_bias = pim.predict(statei, bias, True)
        r, J_local_statej, J_local_statejstar = statej.local(statejstar, True)
        r = r.vec()
        J_statei = J_local_statejstar.dot(J_statejstar_statei)
        J_statej = J_local_statej
        J_biasi = J_local_statejstar.dot(J_statejstar_bias)
        return r, J_statei, J_statej, J_biasi

def to2d(x):
    R = expSO3(x[0:3])
    theta = np.arctan2( R[1,0], R[0,0])
    x2d = np.zeros(3)
    x2d[0:2] = x[3:5]
    x2d[2] = theta
    return x2d
