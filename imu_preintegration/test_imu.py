
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
prevBias = gtsam.imuBias.ConstantBias()
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
    def __init__(self,R,P,V):
        self.R = R
        self.P = P
        self.V = V
    


class imuIntegration:
    def __init__(self):
        self.deltaRij = np.array([0,0,0])
        self.deltaPij = np.array([0,0,0])
        self.deltaVij = np.array([0,0,0])
        self.deltaTij = 0
        self.gravity = np.array([0,0,-G])
        
    def update(self, acc, omega, dt):
        """
        #check imuFactor.pdf: A Simple Euler Scheme (11~13)
        """
        H_inv = np.linalg.inv( H(self.deltaRij))
        Rka = expSO3(self.deltaRij).dot(acc)
        self.deltaRij = self.deltaRij + H_inv.dot(omega) * dt
        self.deltaPij = self.deltaPij + self.deltaVij * dt + Rka*dt*dt/2
        self.deltaVij = self.deltaVij + Rka * dt 
        self.deltaTij += dt

    def predict(self, state):
        """
        #check imuFactor.pdf: Application: The New IMU Factor
        #b: body frame, the local imu frame
        #c: the current imu frame
        #n: the nav frame (world frame)
        """
        dt = self.deltaTij
        dt22 = 0.5 * self.deltaTij * self.deltaTij
        nRb = expSO3(state.R)
        bRn = nRb.T
        bRc = expSO3(self.deltaRij)
        bPc = self.deltaPij + dt * bRn.dot(state.V) + dt22 * bRn.dot(self.gravity)
        bVc = self.deltaVij + dt * bRn.dot(self.gravity)
        nRc = nRb.dot(bRc)
        nPc = state.P + nRb.dot(bPc)
        nVc = state.V + nRb.dot(bVc)
        return navState(logSO3(nRc),nPc,nVc)
imuIntegrator = imuIntegration()

lastImuTime = -1
prevVel = np.array([0,0,0])
prevStateR = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , prevVel)
prevState = navState(np.array([0,0,0]),np.array([0,0,0]),prevVel)
trj0 = []
trj1 = []
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
    #print(imuIntegratorR)
    currStateR = imuIntegratorR.predict(prevStateR, prevBias)
    imuIntegrator.update(i[1:4], i[4:7], dt)
    currState = imuIntegrator.predict(prevState)
    trj0.append([currStateR.pose().translation()[0],currStateR.pose().translation()[1],currStateR.pose().translation()[2]])
    trj1.append([currState.P[0],currState.P[1],currState.P[2]])
print(imuIntegrator.predict(prevState).P)
import matplotlib.pyplot as plt
trj0 = np.array(trj0)
trj1 = np.array(trj1)
plt.plot(trj0[:,0],trj0[:,1],label='trj_ref')
plt.plot(trj1[:,0],trj1[:,1],label='trj_my')
plt.legend()
plt.show()
