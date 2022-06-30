from preintegration import *


class naviNode:
    def __init__(self, state):
        self.state = state
        self.size = 9
    def update(self, dx):
        d_state = navDelta(expSO3(dx[0:3]), dx[3:6], dx[6:9])
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

class velocityEdge:
    def __init__(self, i, j, z, omega = np.eye(3)):
        self.i = i #state i
        self.j = j #state j
        self.z = z #time
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        state_i = nodes[self.i].state
        state_j = nodes[self.j].state
        t_inv = 1./self.z
        r = state_j.v - (state_j.p - state_i.p)*t_inv
        Ji = np.zeros([3,9])
        Jj = np.zeros([3,9])
        Ji[:,3:6] = state_i.R*t_inv
        Jj[:,3:6] = -state_j.R*t_inv
        Jj[:,6:9] = state_j.R
        return r, Ji, Jj


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
        deltaij, J_d_i, J_d_j = state_i.local(state_j,True)
        r, _, J_f_d = self.z.local(deltaij, True)
        r = r.vec()
        Ji = J_f_d.dot(J_d_i)
        Jj = J_f_d.dot(J_d_j)
        return r, Ji, Jj

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

def numericalDerivativeA(func, a, b, z):
    delta = 1e-5
    m = func(a, b, z).vec().shape[0]
    n = a.vec().shape[0]
    J = np.zeros([m,n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        ds = navState()
        ds.set(dx)
        J[:,j] = func(a,b,z).local(func(a.retract(ds),b,z)).vec()/delta
    return J

def numericalDerivativeB(func, a, b, z):
    delta = 1e-5
    m = func(a, b, z).vec().shape[0]
    n = a.vec().shape[0]
    J = np.zeros([m,n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        ds = navState()
        ds.set(dx)
        J[:,j] = func(a,b,z).local(func(a,b.retract(ds),z)).vec()/delta
    return J

if __name__ == '__main__':
    a = navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))
    b = navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.5,0.6,0.7]))
    z = navState(expSO3(np.array([0.3,0.4,0.5])),np.array([0.5,0.6,0.7]),np.array([0.7,0.8,0.9]))


    #Ja = numericalDerivativeA(func,a,b,z)
    #Jb = numericalDerivativeB(func,a,b,z)
    
    