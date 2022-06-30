import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
#from __future__ import annotations
from typing import get_type_hints

class navDelta:
    def __init__(self,R=np.eye(3),p=np.zeros(3,),v=np.zeros(3,)):
        if(R.shape != (3,3) and p.shape != (3,) and v.shape != (3,)):
            print('Set navDelta with a wrong shape.')
            exit(0)
        self.R = R
        self.p = p
        self.v = v

    def vec(self):
        return np.hstack([logSO3(self.R),self.p, self.v])

    def set(self, x):
        self.R = expSO3(x[0:3])
        self.p = x[3:6]
        self.v = x[6:9]

    def retract(self:'navDelta', b: 'navDelta', calc_J = False) ->'navDelta':
        R_j = b.R
        p_j = b.p
        v_j = b.v
        R_i = self.R
        p_i = self.p
        v_i = self.v
        R = R_i.dot(R_j)
        p = p_i + p_j
        v = v_i + v_j
        state = navDelta(R, p, v)
        if(calc_J == False):
            return state
        else:
            J_retract_i = np.eye(9)
            J_retract_i[0:3,0:3] = R_j.T
            J_retract_j = np.eye(9)
            return state, J_retract_i, J_retract_j

    def local(self:'navDelta', state: 'navDelta', calc_J = False) ->'navDelta':
        dR = self.R.T.dot(state.R)
        dp = state.p - self.p
        dv = state.v - self.v
        delta = navState(dR, dp, dv)
        if(calc_J == False):
            return delta
        else:
            J_local_i = -np.eye(9)
            J_local_i[0:3,0:3] = -dR.T
            J_local_j = np.eye(9)
            J_local_j[0:3,0:3] = np.eye(3)
            return delta, J_local_i, J_local_j

class navState:
    def __init__(self,R=np.eye(3),p=np.zeros(3,),v=np.zeros(3,)):
        """
        Check preintegration.md (2)
        The navigation state combined by attitude R, position p and velocity v
        """
        if(R.shape != (3,3) and p.shape != (3,) and v.shape != (3,)):
            print('Set navsate with a wrong shape.')
            exit(0)
        self.R = R
        self.p = p
        self.v = v

    def vec(self):
        return np.hstack([logSO3(self.R),self.p, self.v])

    def set(self, x):
        self.R = expSO3(x[0:3])
        self.p = x[3:6]
        self.v = x[6:9]

    def retract(self:'navState', zeta: navDelta, calc_J = False) -> 'navState':
        """
        Check preintegration.md (22)(23)(24)
        Combine 2 navigation states.
        """
        if(not isinstance(zeta, navDelta)):
            print('zeta must be navDelta.')
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
        state = navState(R_nc, p_nc, v_nc)
        if(calc_J == False):
            return state
        else:
            R_cb = R_bc.T
            J_retract_state = np.eye(9)
            J_retract_state[0:3,0:3] = R_cb
            J_retract_state[3:6,3:6] = R_cb
            J_retract_state[6:9,6:9] = R_cb
            J_retract_state[3:6,0:3] = -R_cb.dot(skew(p_bc))
            J_retract_state[6:9,0:3] = -R_cb.dot(skew(v_bc))
            J_retract_delta = np.eye(9)
            J_retract_delta[3:6,3:6] = R_cb
            J_retract_delta[6:9,6:9] = R_cb
            return state, J_retract_state, J_retract_delta

    def local(self:'navState', state: 'navState', calc_J = False) -> navDelta:
        """
        Check preintegration.md (25)(26)(27)
        Get the difference between 2 navigation states.
        """
        if(not isinstance(state, navState)):
            print('state must be navState.')
            exit(0)
        dR = self.R.T.dot(state.R)
        dp = self.R.T.dot(state.p - self.p)
        dv = self.R.T.dot(state.v - self.v)
        delta = navDelta(dR, dp, dv)
        if(calc_J == False):
            return delta
        else:
            J_local_statei = -np.eye(9)
            J_local_statei[0:3,0:3] = -dR.T
            J_local_statei[3:6,0:3] = skew(dp)
            J_local_statei[6:9,0:3] = skew(dv)
            J_local_statej = np.eye(9)
            J_local_statej[0:3,0:3] = np.eye(3)
            J_local_statej[3:6,3:6] = dR
            J_local_statej[6:9,6:9] = dR
            return delta, J_local_statei, J_local_statej

class navDelta:
    def __init__(self,R=np.eye(3),p=np.zeros(3,),v=np.zeros(3,)):
        if(R.shape != (3,3) and p.shape != (3,) and v.shape != (3,)):
            print('Set navDelta with a wrong shape.')
            exit(0)
        self.R = R
        self.p = p
        self.v = v

    def vec(self):
        return np.hstack([logSO3(self.R),self.p, self.v])

    def set(self, x):
        self.R = expSO3(x[0:3])
        self.p = x[3:6]
        self.v = x[6:9]

    def retract(self, b: navDelta, calc_J = False)-> navDelta:
        if(not isinstance(b, navDelta)):
            print('b must be navDelta.')
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
        state = navDelta(R, p, v)
        if(calc_J == False):
            return state
        else:
            J_retract_i = np.eye(9)
            J_retract_i[0:3,0:3] = R_j.T
            J_retract_j = np.eye(9)
            return state, J_retract_i, J_retract_j

    def local(self, b: navDelta, calc_J = False)-> navDelta:
        if(not isinstance(b, navDelta)):
            print('b must be navDelta.')
            exit(0)
        dR = self.R.T.dot(b.R)
        dp = b.p - self.p
        dv = b.v - self.v
        delta = navState(dR, dp, dv)
        if(calc_J == False):
            return delta
        else:
            J_local_i = -np.eye(9)
            J_local_i[0:3,0:3] = -dR.T
            J_local_j = np.eye(9)
            J_local_j[0:3,0:3] = np.eye(3)
            return delta, J_local_i, J_local_j


class imuIntegration:
    def __init__(self,G):
        self.d_Rij = np.eye(3)
        self.d_pij = np.array([0,0,0])
        self.d_vij = np.array([0,0,0])
        self.d_tij = 0
        self.gravity = np.array([0,0,-G])
        self.J_zeta_bacc = np.zeros([9,3])
        self.J_zeta_bgyo = np.zeros([9,3])

        self.bacc = np.array([0,0,0])
        self.bgyo = np.array([0,0,0])
        self.acc_buf = []
        self.gyo_buf = []
        self.dt_buf = []
        
    def update(self, acc, gyo, dt):
        """
        Check preintegration.md (7)
        Integrates all the IMU measurements without considering IMU bias and the gravity.
        """
        self.acc_buf.append(acc)
        self.gyo_buf.append(gyo)
        self.dt_buf.append(dt)
        acc_unbias = acc - self.bacc
        gyo_unbias = gyo - self.bgyo
        R = self.d_Rij
        Ra = R.dot(acc_unbias)

        self.d_Rij = self.d_Rij.dot(expSO3(gyo_unbias * dt))
        self.d_pij = self.d_pij + self.d_vij * dt + Ra*dt*dt/2
        self.d_vij = self.d_vij + Ra * dt 
        self.d_tij += dt
        """
        Check preintegration.md (8)(9)(10)
        """
        A = np.eye(9)
        dt22 = 0.5 * dt * dt
        Rahat = R.dot(skew(acc_unbias))
        A[0:3,0:3] = np.eye(3) -skew(gyo_unbias) * dt
        A[3:6,0:3] = -Rahat * dt22
        A[3:6,6:9] = np.eye(3) * dt
        A[6:9,0:3] = -Rahat * dt

        B = np.zeros([9,3])
        B[3:6,0:3] = R * dt22
        B[6:9,0:3] = R * dt

        C = np.zeros([9,3])
        C[0:3,0:3] = np.eye(3) * dt
        self.J_zeta_bacc = A.dot(self.J_zeta_bacc) - B
        self.J_zeta_bgyo = A.dot(self.J_zeta_bgyo) - C
        

    def biasCorrect(self, bias, calc_J = False):
        """
        Check preintegration.md (11)~(17)
        Correct the PIM by a given bias.
        """
        bacc_inc = bias[0:3] - self.bacc
        bgyo_inc = bias[3:6] - self.bgyo
        d_xi = self.J_zeta_bacc.dot(bacc_inc) + self.J_zeta_bgyo.dot(bgyo_inc)
        Rij_unbias = self.d_Rij.dot(expSO3(d_xi[0:3]))
        pij_unbias = self.d_pij + d_xi[3:6]
        vij_unbias = self.d_vij + d_xi[6:9]
        xi = navDelta(Rij_unbias,pij_unbias,vij_unbias)
        if(calc_J == False):
            return xi
        else:
            J_xi_bias = np.hstack([self.J_zeta_bacc, self.J_zeta_bgyo])
            return xi, J_xi_bias

    def calcDelta(self, xi, state, calc_J = False):
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
        delta = navDelta(xi.R,p_bc,v_bc)
        if(calc_J == False):
            return delta
        else:
            J_delta_state = np.zeros([9,9])
            J_delta_state[3:6,0:3] = dt * skew(R_bn.dot(v_nb)) + dt22 * skew(R_bn.dot(self.gravity))
            J_delta_state[3:6,6:9] = np.eye(3) * dt
            J_delta_state[6:9,0:3] = dt * skew(R_bn.dot(self.gravity))
            J_delta_xi = np.eye(9)
            return delta, J_delta_state, J_delta_xi


    def predict(self, state, bias, calc_J = False):
        """
        #check imuFactor.pdf: Application: The New IMU Factor
        #b: body frame, the local imu frame
        #c: the current imu frame
        #n: the nav frame (world frame)
        """
        if(calc_J == False):
            xi = self.biasCorrect(bias)
            delta = self.calcDelta(xi, state)
            state_j = state.retract(delta)
            return state_j
        else:
            xi, J_xi_bias = self.biasCorrect(bias, True)
            delta, J_delta_state, J_delta_xi = self.calcDelta(xi, state, True)
            state_j, J_retract_state, J_retract_delta = state.retract(delta, True)
            J_predict_state = J_retract_state + J_retract_delta.dot(J_delta_state)
            J_predict_bias = J_retract_delta.dot(J_delta_xi.dot(J_xi_bias))
            return state_j, J_predict_state, J_predict_bias

def find_nearest(data, stamp):
    idx = (np.abs(data[:,0] - stamp)).argmin()
    return data[idx,:]


if __name__ == '__main__':
    def numericalDerivativeA(func, a, b):
        delta = 1e-8
        m = func(a, b).vec().shape[0]
        n = a.vec().shape[0]
        J = np.zeros([m,n])
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = delta
            dd = navDelta()
            dd.set(dx)
            J[:,j] = func(a,b).local(func(a.retract(dd),b)).vec()/delta
        return J

    def numericalDerivativeB(func, a, b):
        delta = 1e-8
        m = func(a, b).vec().shape[0]
        n = a.vec().shape[0]
        J = np.zeros([m,n])
        for j in range(n):
            dx = np.zeros(n)
            dx[j] = delta
            dd = navDelta()
            dd.set(dx)
            J[:,j] = func(a,b).local(func(a,b.retract(dd))).vec()/delta

        return J
        
    state_i = navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))
    delta = navDelta(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.5,0.6,0.7]))
    print('test state retract')
    r, Ja, Jb = state_i.retract(delta,True)
    Jam =  numericalDerivativeA(navState.retract, state_i, delta)
    Jbm =  numericalDerivativeB(navState.retract, state_i, delta)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')

    print('test state local')
    state_i = navState(expSO3(np.array([0.1,0.2,0.3])),np.array([0.2,0.3,0.4]),np.array([0.4,0.5,0.6]))
    state_j = navState(expSO3(np.array([0.2,0.3,0.4])),np.array([0.4,0.5,0.6]),np.array([0.5,0.6,0.7]))
    r, Ja, Jb = state_i.local(state_j,True)
    Jam =  numericalDerivativeA(navState.local, state_i, state_j)
    Jbm =  numericalDerivativeB(navState.local, state_i, state_j)
    if(np.linalg.norm(Jam - Ja) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(Jbm - Jb) < 0.0001):
        print('OK')
    else:
        print('NG')
    """
    bias = np.array([0.1,0.2,0.3,-0.1,-0.2,-0.3])
    ipi = imuIntegration(9.8)
    ipi.update(np.array([0.1,0.1,0.1]),np.array([0.2,0.2,0.2]),0.1)
    ipi.update(np.array([0.1,0.1,0.1]),np.array([0.2,0.2,0.2]),0.1)
    ipi.update(np.array([0.1,0.1,0.1]),np.array([0.2,0.2,0.2]),0.1)
    state_j, J_predict_state, J_predict_bias = ipi.predict(state_i, bias,True)
    J_predict_state_numerical = numericalDerivativeA(ipi.predict, state_i, bias)
    J_predict_bias_numerical = numericalDerivativeB2(ipi.predict, state_i, bias)
    print('test J_predict_state')
    if(np.linalg.norm(J_predict_state_numerical - J_predict_state) < 0.0001):
        print('OK')
    else:
        print('NG')
    print('test J_predict_bias')
    if(np.linalg.norm(J_predict_bias_numerical - J_predict_bias) < 0.0001):
        print('OK')
    else:
        print('NG')
    """

    
    #Ja = numericalDerivativeA(func,a,b,z)
    #Jb = numericalDerivativeB(func,a,b,z)
    