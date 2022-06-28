import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


class navState:
    def __init__(self,R=np.eye(3),p=np.zeros(3),v=np.zeros(3)):
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

    def retract(self, zeta, calc_J = False):
        """
        Check preintegration.md (22)(23)(24)
        Combine 2 navigation states.
        """
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

    def local(self, state, calc_J = False):
        """
        Check preintegration.md (25)(26)(27)
        Get the difference between 2 navigation states.
        """
        dR = self.R.T.dot(state.R)
        dp = self.R.T.dot(state.p - self.p)
        dv = self.R.T.dot(state.v - self.v)
        delta = navState(dR, dp, dv)
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
        xi = navState(Rij_unbias,pij_unbias,vij_unbias)
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
        delta = navState(xi.R,p_bc,v_bc)
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

