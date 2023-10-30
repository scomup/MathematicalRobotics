import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from scipy.spatial import KDTree


class Vector(np.ndarray):
    def __new__(cls, obj):
        self = np.asarray(obj).view(cls)
        return self

    def vec(self):
        return self

    def retract(self, b):
        return self + b

    def local(self, b):
        return b - self

    def set(x):
        self = Vector(x)
        return self


class NavState:
    def __init__(self, R=np.eye(3), p=np.zeros(3,), v=np.zeros(3,)):
        """
        Check preintegration.md (2)
        The navigation state combined by attitude R, position p and velocity v
        """
        if (R.shape != (3, 3) and p.shape != (3,) and v.shape != (3,)):
            print('Set navsate with a wrong shape.')
            exit(0)
        self.R = R
        self.p = p
        self.v = v

    def vec(self):
        return np.hstack([logSO3(self.R), self.p, self.v])

    def set(x):
        new = NavState()
        new.R = expSO3(x[0:3])
        new.p = x[3:6]
        new.v = x[6:9]
        return new

    def retract(self, zeta: 'NavDelta', calc_J=False) -> 'NavState':
        """
        Check preintegration.md (22)(23)(24)
        Combine 2 navigation states.
        """
        if (not isinstance(zeta, NavDelta)):
            print('zeta must be NavDelta.')
            exit(0)
        R_bc = zeta.R
        p_bc = zeta.p
        v_bc = zeta.v
        R_nb = self.R
        p_nb = self.p
        v_nb = self.v
        R_nc = R_nb.dot(R_bc)
        p_nc = p_nb + R_nb.dot(p_bc)
        v_nc = v_nb + R_nb.dot(v_bc)
        state = NavState(R_nc, p_nc, v_nc)
        if (calc_J is False):
            return state
        else:
            R_cb = R_bc.T
            J_retract_state = np.eye(9)
            J_retract_state[0:3, 0:3] = R_cb
            J_retract_state[3:6, 3:6] = R_cb
            J_retract_state[6:9, 6:9] = R_cb
            J_retract_state[3:6, 0:3] = -R_cb.dot(skew(p_bc))
            J_retract_state[6:9, 0:3] = -R_cb.dot(skew(v_bc))
            J_retract_delta = np.eye(9)
            J_retract_delta[3:6, 3:6] = R_cb
            J_retract_delta[6:9, 6:9] = R_cb
            return state, J_retract_state, J_retract_delta

    def local(self, state: 'NavState', calc_J=False) -> 'NavDelta':
        """
        Check preintegration.md (25)(26)(27)
        Get the difference between 2 navigation states.
        """
        if (not isinstance(state, NavState)):
            print('state must be NavState.')
            exit(0)
        dR = self.R.T.dot(state.R)
        dp = self.R.T.dot(state.p - self.p)
        dv = self.R.T.dot(state.v - self.v)
        delta = NavDelta(dR, dp, dv)
        if (calc_J is False):
            return delta
        else:
            J_local_statei = -np.eye(9)
            J_local_statei[0:3, 0:3] = -dR.T
            J_local_statei[3:6, 0:3] = skew(dp)
            J_local_statei[6:9, 0:3] = skew(dv)
            J_local_statej = np.eye(9)
            J_local_statej[0:3, 0:3] = np.eye(3)
            J_local_statej[3:6, 3:6] = dR
            J_local_statej[6:9, 6:9] = dR
            return delta, J_local_statei, J_local_statej


class NavDelta:
    def __init__(self, R=np.eye(3), p=np.zeros(3,), v=np.zeros(3,)):
        if (R.shape != (3, 3) and p.shape != (3,) and v.shape != (3,)):
            print('Set NavDelta with a wrong shape.')
            exit(0)
        self.R = R
        self.p = p
        self.v = v

    def update(self, acc_unbias, gyo_unbias, dt, calc_J=False):
        R = self.R
        Ra = R.dot(acc_unbias)
        """
        update by the IMU measurement, without considering
        the IMU bias, gravity and initial velocity.
        Check preintegration.md (7)
        """
        R1 = self.R.dot(expSO3(gyo_unbias * dt))
        p1 = self.p + self.v * dt + Ra*dt*dt/2
        v1 = self.v + Ra * dt
        new = NavDelta(R1, p1, v1)
        """
        Check preintegration.md (8)(9)(10)
        """
        if (calc_J is True):
            Jold = np.eye(9)
            dt22 = 0.5 * dt * dt
            Rahat = R.dot(skew(acc_unbias))
            Jold[0:3, 0:3] = expSO3(-gyo_unbias*dt)
            # Jold[0:3, 0:3] = np.eye(3) -skew(gyo_unbias) * dt
            Jold[3:6, 0:3] = -Rahat * dt22
            Jold[3:6, 6:9] = np.eye(3) * dt
            Jold[6:9, 0:3] = -Rahat * dt
            Jacc = np.zeros([9, 3])
            Jacc[3:6, 0:3] = R * dt22
            Jacc[6:9, 0:3] = R * dt
            Jgyo = np.zeros([9, 3])
            Jgyo[0:3, 0:3] = HSO3(gyo_unbias * dt)
            return new, Jold, Jacc, Jgyo
        else:
            return new

    def vec(self):
        return np.hstack([logSO3(self.R), self.p, self.v])

    def set(x):
        new = NavDelta()
        new.R = expSO3(x[0:3])
        new.p = x[3:6]
        new.v = x[6:9]
        return new

    def retract(self, b: 'NavDelta', calc_J=False) -> 'NavDelta':
        if (not isinstance(b, NavDelta)):
            print('b must be NavDelta.')
            exit(0)
        R_j = b.R
        p_j = b.p
        v_j = b.v
        R_i = self.R
        p_i = self.p
        v_i = self.v
        R = R_i.dot(R_j)
        p = p_i + p_j
        v = v_i + v_j
        state = NavDelta(R, p, v)
        if (calc_J is False):
            return state
        else:
            J_retract_i = np.eye(9)
            J_retract_i[0:3, 0:3] = R_j.T
            J_retract_j = np.eye(9)
            return state, J_retract_i, J_retract_j

    def local(self, b: 'NavDelta', calc_J=False) -> 'NavDelta':
        if (not isinstance(b, NavDelta)):
            print('b must be NavDelta.')
            exit(0)
        dR = self.R.T.dot(b.R)
        dp = b.p - self.p
        dv = b.v - self.v
        delta = NavDelta(dR, dp, dv)
        if (calc_J is False):
            return delta
        else:
            J_local_i = -np.eye(9)
            J_local_i[0:3, 0:3] = -dR.T
            J_local_j = np.eye(9)
            J_local_j[0:3, 0:3] = np.eye(3)
            return delta, J_local_i, J_local_j


class ImuIntegration:
    def __init__(self, G, bias=np.zeros(6), Rbi=np.eye(3), tbi=np.zeros(3)):
        self.pim = NavDelta()
        self.d_tij = 0
        self.gravity = np.array([0, 0, -G])
        self.J_zeta_bacc = np.zeros([9, 3])
        self.J_zeta_bgyo = np.zeros([9, 3])
        self.Rbi = Rbi
        self.tbi = tbi
        self.bacc = bias[0:3]
        self.bgyo = bias[3:6]
        self.acc_buf = []
        self.gyo_buf = []
        self.dt_buf = []

    def update(self, acc_i, gyo_i, dt):
        """
        Check preintegration.md (7)
        Integrates all the IMU measurements without considering IMU bias and the gravity.
        """
        acc = self.Rbi.dot(acc_i)
        gyo = self.Rbi.dot(gyo_i)
        acc -= skew(gyo).dot(skew(gyo).dot(self.tbi))
        self.acc_buf.append(acc)
        self.gyo_buf.append(gyo)
        self.dt_buf.append(dt)
        acc_unbias = acc - self.bacc
        gyo_unbias = gyo - self.bgyo
        self.d_tij += dt
        self.pim, Jold, Jacc, Jgyo = self.pim.update(acc_unbias, gyo_unbias, dt, True)
        self.J_zeta_bacc = Jold.dot(self.J_zeta_bacc) - Jacc
        self.J_zeta_bgyo = Jold.dot(self.J_zeta_bgyo) - Jgyo

    def biasCorrect(self, bias, calc_J=False):
        """
        Check preintegration.md (11)~(17)
        Correct the PIM by a given bias.
        """
        bacc_inc = bias[0:3] - self.bacc
        bgyo_inc = bias[3:6] - self.bgyo
        d_xi = self.J_zeta_bacc.dot(bacc_inc) + self.J_zeta_bgyo.dot(bgyo_inc)
        d_state = NavDelta.set(d_xi)
        xi = self.pim.retract(d_state)
        if (calc_J is False):
            return xi
        else:
            J_xi_bias = np.hstack([self.J_zeta_bacc, self.J_zeta_bgyo])
            return xi, J_xi_bias

    def calcDelta(self, xi, state, calc_J=False):
        """
        Check preintegration.md (19)(20)(21)
        Calculate the delta between two navigation states.
        """
        p = xi.p
        v = xi.v
        dt = self.d_tij
        dt22 = 0.5 * self.d_tij * self.d_tij
        R_bn = state.R.T
        v_nb = state.v
        p_bc = p + dt * R_bn.dot(v_nb) + dt22 * R_bn.dot(self.gravity)
        v_bc = v + dt * R_bn.dot(self.gravity)
        delta = NavDelta(xi.R, p_bc, v_bc)
        if (calc_J is False):
            return delta
        else:
            J_delta_state = np.zeros([9, 9])
            J_delta_state[3:6, 0:3] = dt * skew(R_bn.dot(v_nb)) + dt22 * skew(R_bn.dot(self.gravity))
            J_delta_state[3:6, 6:9] = np.eye(3) * dt
            J_delta_state[6:9, 0:3] = dt * skew(R_bn.dot(self.gravity))
            J_delta_xi = np.eye(9)
            return delta, J_delta_xi, J_delta_state

    def predict(self, state, bias, calc_J=False):
        """
        # check imuFactor.pdf: Application: The New IMU Factor
        # b: body frame, the local imu frame
        # c: the current imu frame
        # n: the nav frame (world frame)
        """
        if (calc_J is False):
            xi = self.biasCorrect(bias)
            delta = self.calcDelta(xi, state)
            state_j = state.retract(delta)
            return state_j
        else:
            xi, J_xi_bias = self.biasCorrect(bias, True)
            delta, J_delta_xi, J_delta_state = self.calcDelta(xi, state, True)
            state_j, J_retract_state, J_retract_delta = state.retract(delta, True)
            J_predict_state = J_retract_state + J_retract_delta.dot(J_delta_state)
            J_predict_bias = J_retract_delta.dot(J_delta_xi.dot(J_xi_bias))
            return state_j, J_predict_state, J_predict_bias


def find_nearest(data, stamp):
    idx = (np.abs(data[:, 0] - stamp)).argmin()
    return data[idx, :].copy()


class FindNearest3D:
    def __init__(self, data) -> None:
        self.data = data
        self.data3d = data[:, 1:4]
        self.kdtree = KDTree(self.data3d)

    def query(self, sample):
        dist, point = self.kdtree.query(sample, 1)
        return dist, point


if __name__ == '__main__':
    def numericalDerivative(func, param, idx, TYPE=None):
        if TYPE is None:
            TYPE = type(param[idx])
        delta = 1e-5
        m = func(*param).vec().shape[0]
        n = (param[idx]).vec().shape[0]
        J = np.zeros([m, n])
        h = func(*param)
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = delta
            dd = TYPE.set(dx)
            param_delta = param.copy()
            param_delta[idx] = param[idx].retract(dd)
            h_plus = func(*param_delta)
            J[:, j] = h.local(h_plus).vec()/delta
        return J

    state_i = NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    delta = NavDelta(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.6, 0.7]))
    print('test state retract')
    r, Ja, Jb = state_i.retract(delta, True)
    Jam = numericalDerivative(NavState.retract, [state_i, delta], 0, NavDelta)
    Jbm = numericalDerivative(NavState.retract, [state_i, delta], 1)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test state local')
    state_i = NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    state_j = NavState(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.6, 0.7]))
    r, Ja, Jb = state_i.local(state_j, True)
    Jam = numericalDerivative(NavState.local, [state_i, state_j], 0, NavDelta)
    Jbm = numericalDerivative(NavState.local, [state_i, state_j], 1, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')
    delta_i = NavDelta(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    delta_j = NavDelta(expSO3(np.array([0.2, 0.3, 0.4])), np.array([0.4, 0.5, 0.6]), np.array([0.5, 0.6, 0.7]))
    print('test delta retract')
    r, Ja, Jb = delta_i.retract(delta_j, True)
    Jam = numericalDerivative(NavDelta.retract, [delta_i, delta_j], 0)
    Jbm = numericalDerivative(NavDelta.retract, [delta_i, delta_j], 1)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test delta local')
    r, Ja, Jb = delta_i.local(delta_j, True)
    Jam = numericalDerivative(NavDelta.local, [delta_i, delta_j], 0)
    Jbm = numericalDerivative(NavDelta.local, [delta_i, delta_j], 1)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test pim update')
    delta_i = NavDelta(expSO3(np.array([0.5, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    acc = Vector([0.01, 0.02, 0.03])
    gyo = Vector([1.01, 0.5, 0.23])
    dt = 1.
    r, Jold, Jacc, Jgyo = delta_i.update(acc, gyo, dt, True)
    Joldm = numericalDerivative(NavDelta.update, [delta_i, acc, gyo, dt], 0)
    Jaccm = numericalDerivative(NavDelta.update, [delta_i, acc, gyo, dt], 1)
    Jgyom = numericalDerivative(NavDelta.update, [delta_i, acc, gyo, dt], 2)
    if (np.linalg.norm(Joldm - Jold) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jaccm - Jacc) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jgyom - Jgyo) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test delta')
    xi = NavDelta(expSO3(np.array([0.5, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    state_i = NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))
    imu = ImuIntegration(9.8)
    imu.d_tij = 1
    r, Ja, Jb = imu.calcDelta(xi, state_i, True)

    Jam = numericalDerivative(imu.calcDelta, [xi, state_i], 0)
    Jbm = numericalDerivative(imu.calcDelta, [xi, state_i], 1, NavDelta)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test biasCorrect')
    bias = Vector([0.11, 0.12, 0.01, 0.2, 0.15, 0.16])
    imu = ImuIntegration(9.8, bias)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    state_i = NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))

    xi, Ja = imu.biasCorrect(bias, True)
    Jam = numericalDerivative(imu.biasCorrect, [bias], 0)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test predict')
    bias = Vector([0.11, 0.12, 0.01, 0.2, 0.15, 0.16])
    imu = ImuIntegration(9.8, bias)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    imu.update(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]), 0.1)
    state_i = NavState(expSO3(np.array([0.1, 0.2, 0.3])), np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]))

    state_j, Ja, Jb = imu.predict(state_i, bias, True)
    Jam = numericalDerivative(imu.predict, [state_i, bias], 0, NavDelta)
    Jbm = numericalDerivative(imu.predict, [state_i, bias], 1)
    if (np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if (np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')
