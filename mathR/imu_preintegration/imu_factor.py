import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from imu_preintegration.preintegration import *
from graph_optimization.graph_solver import BaseVertex, BaseEdge


class NaviVertex(BaseVertex):
    def __init__(self, x, stamp=0, id=0):
        super().__init__(x, 9)
        self.stamp = stamp
        self.id = id

    def update(self, dx):
        d_state = NavDelta(expSO3(dx[0:3]), dx[3:6], dx[6:9])
        self.x = self.x.retract(d_state)


class BiasVertex(BaseVertex):
    def __init__(self, x, id=0):
        super().__init__(x, 6)
        self.id = id

    def update(self, dx):
        self.x = self.x + dx


class BiasEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6)):
        super().__init__(link, z, omega, kernel=None)

    def residual(self, vertices):
        bias_i = vertices[self.link[0]].x
        r = bias_i - self.z
        return r, [np.eye(6)]


class BiasChangeEdge(BaseEdge):
    def __init__(self, link, omega=np.eye(6)):
        super().__init__(link, None, omega, kernel=None)

    def residual(self, vertices):
        bias_i = vertices[self.link[0]].x
        bias_j = vertices[self.link[1]].x
        r = bias_i - bias_j
        return r, [np.eye(6), -np.eye(6)]


class NaviEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(9)):
        super().__init__(link, z, omega, kernel=None)

    def residual(self, vertices):
        state = vertices[self.link[0]].x
        r, j, _ = state.local(self.z, True)
        r = r.vec()
        return r, [j]


class PosvelEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(3)):
        super().__init__(link, z, omega, kernel=None)

    def residual(self, vertices):
        state_i = vertices[self.link[0]].x
        state_j = vertices[self.link[1]].x
        t_inv = 1./self.z
        r = state_i.v - (state_j.p - state_i.p)*t_inv
        Ji = np.zeros([3, 9])
        Jj = np.zeros([3, 9])
        Ji[:, 3:6] = state_i.R*t_inv
        Ji[:, 6:9] = state_i.R
        Jj[:, 3:6] = -state_j.R*t_inv
        return r, [Ji, Jj]


class NavitransEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(9)):
        super().__init__(link, z, omega, kernel=None)

    def residual(self, vertices):
        state_i = vertices[self.link[0]].x
        state_j = vertices[self.link[1]].x
        deltaij, J_d_i, J_d_j = state_i.local(state_j, True)
        r, _, J_f_d = self.z.local(deltaij, True)
        r = r.vec()
        Ji = J_f_d.dot(J_d_i)
        Jj = J_f_d.dot(J_d_j)
        return r, [Ji, Jj]


class ImupreintEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(9)):
        super().__init__(link, z, omega, kernel=None)

    def residual(self, vertices):
        pim = self.z
        statei = vertices[self.link[0]].x
        statej = vertices[self.link[1]].x
        bias = vertices[self.link[2]].x
        statejstar, J_statejstar_statei, J_statejstar_bias = pim.predict(statei, bias, True)
        r, J_local_statej, J_local_statejstar = statej.local(statejstar, True)
        r = r.vec()
        J_statei = J_local_statejstar.dot(J_statejstar_statei)
        J_statej = J_local_statej
        J_biasi = J_local_statejstar.dot(J_statejstar_bias)
        return r, [J_statei, J_statej, J_biasi]


def to2d(x):
    R = expSO3(x[0:3])
    theta = np.arctan2(R[1, 0], R[0, 0])
    x2d = np.zeros(3)
    x2d[0:2] = x[3:5]
    x2d[2] = theta
    return x2d


if __name__ == '__main__':
    import copy

    def numericalDerivative(func, param, idx, TYPE_INPUT=None, TYPE_RETURN=None):
        if TYPE_INPUT is None:
            TYPE_INPUT = type(param[idx].x)
        delta = 1e-5
        m = func(param)[0].shape[0]
        n = (param[idx]).x.vec().shape[0]
        J = np.zeros([m, n])
        h = TYPE_RETURN.set(func(param)[0])
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = delta
            dd = TYPE_INPUT.set(dx)
            param_delta = copy.deepcopy(param)
            param_delta[idx].x = param[idx].x.retract(dd)
            h_plus = TYPE_RETURN.set(func(param_delta)[0])
            J[:, j] = h.local(h_plus).vec()/delta
        return J

    print('test ImupreintEdge')
    bias = Vector([0.11, 0.12, 0.01, 0.2, 0.15, 0.16])
    imu = ImuIntegration(9.8, bias)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    vertices = []
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))))
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.1, 0.2, 0.3]))))
    vertices.append(BiasVertex(Vector([0.11, 0.12, 0.01, 0.2, 0.15, 0.16])))
    edge = ImupreintEdge([0, 1, 2], imu)
    r, J = edge.residual(vertices)
    Ja, Jb, Jc = J

    Jam = numericalDerivative(edge.residual, vertices, 0, NavDelta, NavDelta)
    Jbm = numericalDerivative(edge.residual, vertices, 1, NavDelta, NavDelta)
    Jcm = numericalDerivative(edge.residual, vertices, 2, Vector, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jcm - Jc) < 0.0001):
            print('OK')
    else:
        print('NG')

    print('test NavitransEdge')
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))))
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.1, 0.2, 0.3]))))
    z = NavDelta(expSO3(np.array([0.5, 0.6, 0.7])), np.array([0.1, 0.2, 0.3]), np.array([-0.1, -0.2, -0.3]))

    edge = NavitransEdge([0, 1], z)
    r, J = edge.residual(vertices)
    Ja, Jb = J
    Jam = numericalDerivative(edge.residual, vertices, 0, NavDelta, NavDelta)
    Jbm = numericalDerivative(edge.residual, vertices, 1, NavDelta, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test NavitransEdge')
    vertices = []
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))))
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.1, 0.2, 0.3]))))
    z = NavDelta(expSO3(np.array([0.5, 0.6, 0.7])), np.array([0.1, 0.2, 0.3]), np.array([-0.1, -0.2, -0.3]))

    edge = NavitransEdge([0, 1], z)
    r, J = edge.residual(vertices)
    Ja, Jb = J
    Jam = numericalDerivative(edge.residual, vertices, 0, NavDelta, NavDelta)
    Jbm = numericalDerivative(edge.residual, vertices, 1, NavDelta, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test PosvelEdge')
    edge = PosvelEdge([0, 1], 1)
    r, J = edge.residual(vertices)
    Ja, Jb = J
    Jam = numericalDerivative(edge.residual, vertices, 0, NavDelta, Vector)
    Jbm = numericalDerivative(edge.residual, vertices, 1, NavDelta, Vector)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test NaviEdge')
    vertices = []
    vertices.append(NaviVertex(
        NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))))
    z = NavState(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.1, 0.2, 0.3]))
    edge = NaviEdge([0], z)
    r, J = edge.residual(vertices)
    Ja = J[0]
    Jam = numericalDerivative(edge.residual, vertices, 0, NavDelta, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
