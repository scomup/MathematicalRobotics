from preintegration import *


class naviNode:
    def __init__(self, state, stamp = 0):
        self.x = state
        self.size = 9
        self.stamp = stamp
    def update(self, dx):
        d_state = navDelta(expSO3(dx[0:3]), dx[3:6], dx[6:9])
        self.x = self.x.retract(d_state)

class biasNode:
    def __init__(self, bias):
        self.x = bias
        self.size = 6
    def update(self, dx):
        self.x = self.x + dx

class biasEdge:
    def __init__(self, i, z, omega = np.eye(6)):
        self.i = i #bias i
        self.type = 'one'
        self.z = z
        self.omega = omega
    def residual(self, nodes):
        bias_i = nodes[self.i].x
        r = bias_i - self.z 
        return r, np.eye(6)

class biaschangeEdge:
    def __init__(self, i, j, omega = np.eye(6)):
        self.i = i #bias i
        self.j = j #bias j
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        bias_i = nodes[self.i].x
        bias_j = nodes[self.j].x
        r = bias_i - bias_j 
        return r, np.eye(6), -np.eye(6)

class naviEdge:
    def __init__(self, i, z, omega = np.eye(9)):
        self.i = i
        self.z = z
        self.type = 'one'
        self.omega = omega
    def residual(self, nodes):
        state = nodes[self.i].x
        r, j, _ = state.local(self.z, True)
        r = r.vec()
        return r, j

class posvelEdge:
    def __init__(self, i, j, z, omega = np.eye(3)):
        self.i = i #state i
        self.j = j #state j
        self.z = z #time
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        state_i = nodes[self.i].x
        state_j = nodes[self.j].x
        t_inv = 1./self.z
        r = state_i.v - (state_j.p - state_i.p)*t_inv
        Ji = np.zeros([3,9])
        Jj = np.zeros([3,9])
        Ji[:,3:6] = state_i.R*t_inv
        Ji[:,6:9] = state_i.R
        Jj[:,3:6] = -state_j.R*t_inv
        return r, Ji, Jj


class navitransEdge:
    def __init__(self, i, j, z, omega = np.eye(9)):
        self.i = i #state i
        self.j = j #state j
        self.z = z #error between state i and j
        self.type = 'two'
        self.omega = omega
    def residual(self, nodes):
        state_i = nodes[self.i].x
        state_j = nodes[self.j].x
        deltaij, J_d_i, J_d_j = state_i.local(state_j,True)
        r, _, J_f_d = self.z.local(deltaij, True)
        r = r.vec()
        Ji = J_f_d.dot(J_d_i)
        Jj = J_f_d.dot(J_d_j)
        return r, Ji, Jj

class imupreintEdge:
    def __init__(self, i, j, k, z, omega = np.eye(9)):
        self.i = i #state i
        self.j = j #state j
        self.k = k #bias i
        self.z = z #pim between ij
        self.type = 'three'
        self.omega = omega
    def residual(self, nodes):
        pim = self.z
        statei = nodes[self.i].x
        statej = nodes[self.j].x
        bias = nodes[self.k].x
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


if __name__ == '__main__':
    import copy
    def numericalDerivative(func, param, idx, TYPE_INPUT = None, TYPE_RETURN = None):
        if TYPE_INPUT is None:
            TYPE_INPUT = type(param[idx].x)
        delta = 1e-5
        m = func(param)[0].shape[0]
        n = (param[idx]).x.vec().shape[0]
        J = np.zeros([m,n])
        h = TYPE_RETURN.set(func(param)[0])
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = delta
            dd = TYPE_INPUT.set(dx)
            param_delta = copy.deepcopy(param)
            param_delta[idx].x = param[idx].x.retract(dd)
            h_plus = TYPE_RETURN.set(func(param_delta)[0])
            J[:,j] = h.local(h_plus).vec()/delta
        return J

    print('test imupreintEdge')
    bias = Vector([0.11,0.12,0.01,0.2,0.15,0.16])
    imu = imuIntegration(9.8, bias)
    imu.update(np.array([0.1,0.2,0.3]),np.array([0.1,0.2,0.3]),0.1)
    imu.update(np.array([0.1,0.2,0.3]),np.array([0.1,0.2,0.3]),0.1)
    imu.update(np.array([0.1,0.2,0.3]),np.array([0.1,0.2,0.3]),0.1)
    nodes = []
    nodes.append(naviNode(navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))))
    nodes.append(naviNode(navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.1,0.2,0.3]))))
    nodes.append(biasNode(Vector([0.11,0.12,0.01,0.2,0.15,0.16])))
    edge = imupreintEdge(0,1,2,imu)
    r,Ja,Jb,Jc = edge.residual(nodes)


    Jam =  numericalDerivative(edge.residual, nodes, 0, navDelta, navDelta)
    Jbm =  numericalDerivative(edge.residual, nodes, 1, navDelta, navDelta)
    Jcm =  numericalDerivative(edge.residual, nodes, 2, Vector, navDelta)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jcm - Jc) < 0.0001):
            print('OK')
    else:
        print('NG')
    
    print('test navitransEdge')
    nodes.append(naviNode(navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))))
    nodes.append(naviNode(navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.1,0.2,0.3]))))
    z = navDelta(expSO3(np.array([0.5,0.6,0.7])),np.array([0.1,0.2,0.3]),np.array([-0.1,-0.2,-0.3]))
    
    edge = navitransEdge(0,1,z)
    r,Ja,Jb = edge.residual(nodes)
    Jam =  numericalDerivative(edge.residual, nodes, 0, navDelta, navDelta)
    Jbm =  numericalDerivative(edge.residual, nodes, 1, navDelta, navDelta)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test navitransEdge')
    nodes = []
    nodes.append(naviNode(navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))))
    nodes.append(naviNode(navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.1,0.2,0.3]))))
    z = navDelta(expSO3(np.array([0.5,0.6,0.7])),np.array([0.1,0.2,0.3]),np.array([-0.1,-0.2,-0.3]))
    
    edge = navitransEdge(0,1,z)
    r,Ja,Jb = edge.residual(nodes)
    Jam =  numericalDerivative(edge.residual, nodes, 0, navDelta, navDelta)
    Jbm =  numericalDerivative(edge.residual, nodes, 1, navDelta, navDelta)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test posvelEdge')
    edge = posvelEdge(0,1,1)
    r,Ja,Jb = edge.residual(nodes)
    Jam =  numericalDerivative(edge.residual, nodes, 0, navDelta, Vector)
    Jbm =  numericalDerivative(edge.residual, nodes, 1, navDelta, Vector)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test naviEdge')
    nodes = []
    nodes.append(naviNode(navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))))
    z = navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.1,0.2,0.3]))
    edge = naviEdge(0,z)
    r,Ja = edge.residual(nodes)
    Jam =  numericalDerivative(edge.residual, nodes, 0, navDelta, navDelta)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')


