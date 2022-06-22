
from cProfile import label
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X
import sys,os

from sklearn.metrics import zero_one_loss
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
G = 9.8
imu_params = gtsam.PreintegrationParams.MakeSharedU(G)
prevBias = gtsam.imuBias.ConstantBias(np.array([0,0,0]),np.array([0,0,0]))
imuIntegratorR = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias(np.array([0.007,-0.038,-0.001]),np.array([0.,0.,0.])))

imu = np.load('/home/liu/bag/warehouse/b2_imu.npy')

#check imuFactor.pdf: Derivative of The Local Coordinate Mapping
def H(theta):
    h = np.eye(3)
    theta_shew = skew(theta)
    theta_shew_k = np.eye(3)
    m = 1
    n = 1
    for k in range(1,20):
        m *= (k+1)
        n *= -1
        theta_shew_k = theta_shew_k.dot(theta_shew)
        h += (n/m)*theta_shew_k
    return h

class navState:
    def __init__(self,theta,p,v):
        if(theta.shape==(3,)):
            self.R = expSO3(theta)
        elif(theta.shape==(3,3)):
            self.R = theta
        self.p = p
        self.v = v
    def retract(self, zeta, calc_J = False):
        R_bc = expSO3(zeta[0:3])
        p_bc = zeta[3:6]
        v_bc = zeta[6:9]
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
            J_retract_state = np.zeros([9,9])
            J_retract_state[0:3,0:3] = R_cb
            J_retract_state[3:6,3:6] = R_cb
            J_retract_state[6:9,6:9] = R_cb
            J_retract_state[3:6,0:3] = R_cb.dot(skew(-p_bc))
            J_retract_state[6:9,0:3] = R_cb.dot(skew(-v_bc))
            J_retract_delta = np.zeros([9,9])
            J_retract_delta[0:3,0:3] = HSO3(logSO3(R_bc))
            J_retract_delta[3:6,3:6] = R_cb
            J_retract_delta[6:9,6:9] = R_cb
            return state, J_retract_state, J_retract_delta

    def local(self, state, calc_J = False):
        dR = self.R.T.dot(state.R)
        dp = self.R.T.dot(state.p - self.p)
        dv = self.R.T.dot(state.v - self.v)
        dtheta = logSO3(dR)
        delta = np.hstack([dtheta, dp, dv])
        if(calc_J == False):
            return delta
        else:
            dlog = dLogSO3(dtheta)
            J_local_statei = -np.eye(9)
            J_local_statei[0:3,0:3] = dlog.dot(-dR.T)
            J_local_statei[3:6,0:3] = skew(dp)
            J_local_statei[6:9,0:3] = skew(dv)
            J_local_statej = np.eye(9)
            J_local_statej[0:3,0:3] = dlog
            J_local_statej[3:6,3:6] = dR
            J_local_statej[6:9,6:9] = dR
            return delta, J_local_statei, J_local_statej

class imuIntegration:
    def __init__(self):
        self.d_thetaij = np.array([0,0,0])
        self.d_pij = np.array([0,0,0])
        self.d_vij = np.array([0,0,0])
        self.d_tij = 0
        self.gravity = np.array([0,0,-G])
        self.J_zeta_bacc = np.zeros([9,3])
        self.J_zeta_bgyo = np.zeros([9,3])

        self.bacc = np.array([0,0,0])
        self.bgyo = np.array([0,0,0])
        
    def update(self, acc, gyo, dt):
        """
        #check imuFactor.pdf: A Simple Euler Scheme (11~13)
        """
        acc_unbias = acc - self.bacc
        gyo_unbias = gyo - self.bgyo
        R = expSO3(self.d_thetaij)
        H = HSO3(self.d_thetaij)
        H_inv = np.linalg.inv(H)
        Ra = R.dot(acc_unbias)
        dHinv = dHinvSO3(self.d_thetaij, gyo_unbias)
        #dHinv = -skew(gyo_unbias) * 0.5

        self.d_thetaij = self.d_thetaij + H_inv.dot(gyo_unbias) * dt
        self.d_pij = self.d_pij + self.d_vij * dt + Ra*dt*dt/2
        self.d_vij = self.d_vij + Ra * dt 
        self.d_tij += dt
        
        A = np.eye(9)
        dt22 = 0.5 * dt * dt
        a_nav_H_theta = R.dot(skew(acc_unbias)).dot(H)
        A[0:3,0:3] = np.eye(3) + dHinv * dt
        A[3:6,0:3] = -a_nav_H_theta * dt22
        A[3:6,6:9] = np.eye(3) * dt
        A[6:9,0:3] = -a_nav_H_theta * dt

        B = np.zeros([9,3])
        B[3:6,0:3] = R * dt22
        B[6:9,0:3] = R * dt

        C = np.zeros([9,3])
        C[0:3,0:3] = H_inv * dt
        self.J_zeta_bacc = A.dot(self.J_zeta_bacc) - B
        self.J_zeta_bgyo = A.dot(self.J_zeta_bgyo) - C
        

    def biasCorrect(self, bias, calc_J = False):
        bacc_inc = bias[0:3] - self.bacc
        bgyo_inc = bias[3:6] - self.bgyo
        zeta = np.hstack([self.d_thetaij,self.d_pij,self.d_vij ])
        xi = zeta + self.J_zeta_bacc.dot(bacc_inc) + self.J_zeta_bgyo.dot(bgyo_inc)
        if(calc_J == False):
            return xi
        else:
            J_xi_bias = np.hstack([self.J_zeta_bacc, self.J_zeta_bgyo])
            return xi, J_xi_bias

    def calcDelta(self, xi, state, calc_J = False):
        p = xi[3:6]
        v = xi[6:9]
        dt = self.d_tij
        dt22 = 0.5 * self.d_tij * self.d_tij
        R_bn = state.R.T
        v_nb = state.v
        p_bc = p + dt * R_bn.dot(v_nb) + dt22 * R_bn.dot(self.gravity)
        v_bc = v + dt * R_bn.dot(self.gravity)
        delta = np.hstack([xi[0:3], p_bc, v_bc])
        if(calc_J == False):
            return delta
        else:
            J_delta_state = np.zeros([9,9])
            J_delta_state[3:6,0:3] = dt * skew(R_bn.dot(v_nb)) + dt22 * skew(R_bn.dot(self.gravity))
            J_delta_state[3:6,6:9] = np.eye(3) * dt
            J_delta_state[6:9,0:3] = dt * skew(R_bn.dot(self.gravity))
            J_delta_xi = np.eye(9)
            return delta, J_delta_state, J_delta_xi


    def predict(self, state, bias):
        """
        #check imuFactor.pdf: Application: The New IMU Factor
        #b: body frame, the local imu frame
        #c: the current imu frame
        #n: the nav frame (world frame)
        """
        xi, J_xi_bias = self.biasCorrect(bias, True)
        delta, J_delta_state, J_delta_xi = self.calcDelta(xi, state, True)
        state_j, J_retract_state, J_retract_delta = state.retract(delta, True)
        J_predict_state = J_retract_state + J_retract_delta.dot(J_delta_state)
        J_predict_bias = J_retract_delta.dot(J_delta_xi.dot(J_xi_bias))
        return state_j, J_predict_state, J_predict_bias

if __name__ == '__main__':
    si = navState(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4]), np.array([0.3, 0.3, 0.3]))
    sj = navState(np.array([0.2, 0.3, 0.4]), np.array([0.4, 0.5, 0.6]), np.array([0.4, 0.4, 0.4]))
    ds,H1,H2 = si.local(sj, True)
    #exit(0)
    imuIntegrator = imuIntegration()
    acc = np.array([0.1, 0.2, 10])
    omega = np.array([0.1, 0.2, 0.3])
    imuIntegrator.update(acc, omega, 1)
    imuIntegrator.update(acc, omega, 1)
    imuIntegrator.update(acc, omega, 1)
    state_i = navState(np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]))
    state_j_star = imuIntegrator.predict(state_i,np.array([0.,0.,0.,0.,0.,0.]))
    exit(0)
imuIntegrator = imuIntegration()
lastImuTime = -1
prevVel = np.array([0,0,0])
prevStateR = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , prevVel)
prevStateRR = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , prevVel)
prevState = navState(np.array([0,0,0]),np.array([0,0,0]),prevVel)

trj0 = []
trj1 = []
trj2 = []
for i in imu:
    imuTime = i[0]
    dt = 0
    if(lastImuTime < 0):
        dt = 0.01
    else:
        dt = imuTime - lastImuTime
    if dt <= 0:
        continue
    imuIntegratorR.integrateMeasurement(i[1:4], i[4:7], dt)
    currStateR = imuIntegratorR.predict(prevStateR, gtsam.imuBias.ConstantBias(np.array([0.01,0.01,0.01]),np.array([0.0,0.0,0.0])))
    imuIntegrator.update(i[1:4], i[4:7], dt)
    currState = imuIntegrator.predict(prevState,np.array([0.01,0.01,0.01,0.0,0.0,0.0]))
    trj0.append([currStateR.pose().translation()[0],currStateR.pose().translation()[1],currStateR.pose().translation()[2]])
    trj1.append([currState.p[0],currState.p[1],currState.p[2]])
    #if(imuIntegrator.d_tij > 20):
    #    break
#print(imuIntegrator.predict(prevState).P)
import matplotlib.pyplot as plt
trj0 = np.array(trj0)
trj1 = np.array(trj1)
plt.plot(trj0[:,0],trj0[:,1],label='trj_ref')
plt.plot(trj1[:,0],trj1[:,1],label='trj_my')
plt.legend()
plt.show()
