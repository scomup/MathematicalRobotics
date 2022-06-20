
from cProfile import label
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import B, V, X
import sys,os
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
        self.theta = theta
        self.p = p
        self.v = v
    


class imuIntegration:
    def __init__(self):
        self.d_thetaij = np.array([0,0,0])
        self.d_pij = np.array([0,0,0])
        self.d_vij = np.array([0,0,0])
        self.d_tij = 0
        self.gravity = np.array([0,0,-G])
        self.D_zeta_bacc = np.zeros([9,3])
        self.D_zeta_bgyo = np.zeros([9,3])
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
        #dHinv = dHinvSO3(self.d_thetaij, gyo_unbias)
        dHinv = -skew(gyo_unbias) * 0.5

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
        self.D_zeta_bacc = A.dot(self.D_zeta_bacc) - B
        self.D_zeta_bgyo = A.dot(self.D_zeta_bgyo) - C
        #self.D_zeta_bgyo[0:3,0:3] = self.d_tij * np.eye(3)
        

    def biasCorrect(self, bias):
        bacc_inc = bias[0:3] - self.bacc
        bgyo_inc = bias[3:6] - self.bgyo
        pim = np.hstack([self.d_thetaij,self.d_pij,self.d_vij ])
        pim_unbias = pim + self.D_zeta_bacc.dot(bacc_inc) + self.D_zeta_bgyo.dot(bgyo_inc)
        return pim_unbias[0:3],pim_unbias[3:6],pim_unbias[6:9]

    def predict(self, state, bias):
        """
        #check imuFactor.pdf: Application: The New IMU Factor
        #b: body frame, the local imu frame
        #c: the current imu frame
        #n: the nav frame (world frame)
        """
        d_thetaij, d_pij, d_vij = self.biasCorrect(bias)
        dt = self.d_tij
        dt22 = 0.5 * self.d_tij * self.d_tij
        R_nb = expSO3(state.theta)
        R_bn = R_nb.T
        R_bc = expSO3(d_thetaij)
        p_bc = d_pij + dt * R_bn.dot(state.v) + dt22 * R_bn.dot(self.gravity)
        v_bc = d_vij + dt * R_bn.dot(self.gravity)
        R_nc = R_nb.dot(R_bc)
        p_nc = state.p + R_nb.dot(p_bc)
        v_nc = state.v + R_nb.dot(v_bc)
        return navState(logSO3(R_nc),p_nc,v_nc)
        
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
    if(imuIntegrator.d_tij > 20):
        break
#print(imuIntegrator.predict(prevState).P)
import matplotlib.pyplot as plt
trj0 = np.array(trj0)
trj1 = np.array(trj1)
plt.plot(trj0[:,0],trj0[:,1],label='trj_ref')
plt.plot(trj1[:,0],trj1[:,1],label='trj_my')
plt.legend()
plt.show()
