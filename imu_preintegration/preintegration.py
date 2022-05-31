
from typing import Optional, Sequence

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.utils.plot import plot_pose3
from mpl_toolkits.mplot3d import Axes3D
from gtsam.symbol_shorthand import B, V, X
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from utilities.math_tools import *
from graph_optimization.plot_pose import *

IMU_FIG = 1
POSES_FIG = 2
GRAVITY = 9.81


pose_data = np.load('/home/liu/bag/pose0.npy') 
imu_data = np.load('/home/liu/bag/imu0.npy')


imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
imu_params.setAccelerometerCovariance(np.eye(3) * np.power( 0.01, 2))
imu_params.setIntegrationCovariance(np.eye(3) * np.power( 0, 2))
imu_params.setGyroscopeCovariance(np.eye(3) * np.power(  0.00175, 2))
imu_params.setOmegaCoriolis(np.zeros(3))
imuIntegratorOpt = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias())

priorPoseNoise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) ) # rad,rad,rad,m, m, m
priorVelNoise = gtsam.noiseModel.Isotropic.Sigma(3, 100) 
priorBiasNoise = gtsam.noiseModel.Isotropic.Sigma(6, 10) 
correctionNoise = gtsam.noiseModel.Diagonal.Sigmas(np.array([ 0.05, 0.05, 0.05, 1, 1, 1]))

optParameters = gtsam.ISAM2Params()
optParameters.setRelinearizeThreshold(0.1)
optimizer = gtsam.ISAM2(optParameters)
graphFactors = gtsam.NonlinearFactorGraph()
graphValues = gtsam.Values()

imuAccBiasN = 6.4356659353532566e-10
imuGyrBiasN = 3.5640318696367613e-10
noiseModelBetweenBias = np.array([imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN])


key = 0
begin_idx = 0
begin_time = 0
imus = []
n= pose_data.shape[0]
#n= 5
for i in range(n-1):
#for i in range(2):
    p0 = pose_data[i]
    curPose = gtsam.Pose3(gtsam.Rot3.Quaternion(*p0[4:8]), gtsam.Point3(*p0[1:4]))
    print(key)
    print(curPose)
    p1 = pose_data[i+1]
    dt = p1[0] - p0[0]
    graphValues.insert(X(key), curPose)
    graphValues.insert(V(key), (p1[1:4] - p0[1:4])/dt)
    graphValues.insert(B(key), gtsam.imuBias.ConstantBias())
    key += 1
    tmp = []
    for j in imu_data[begin_idx:]:
        imuTime = j[0]
        begin_idx += 1
        if(imuTime< p0[0] ):
            continue
        if(imuTime > p1[0]):
            break
        if dt <= 0:
            continue
        tmp.append(j)
    imus.append(tmp)

graphFactors.add(gtsam.PriorFactorPose3(X(0), graphValues.atPose3(X(0)), priorPoseNoise))
graphFactors.add(gtsam.PriorFactorPoint3(V(0), graphValues.atPoint3(V(0)), priorVelNoise))
graphFactors.add(gtsam.PriorFactorConstantBias(B(0), graphValues.atConstantBias(B(0)), priorBiasNoise))


begin_idx = 0
for key in range(n-2):
    pass
    imu_slide = imus[i]
    imuIntegratorOpt.resetIntegrationAndSetBias(gtsam.imuBias.ConstantBias())
    for imu in imu_slide:
        imuIntegratorOpt.integrateMeasurement(imu[1:4], imu[4:7], 0.01)
    imu_factor = gtsam.ImuFactor(X(key), V(key), X(key+1), V(key+1), B(key), imuIntegratorOpt)
    bias_factor = gtsam.BetweenFactorConstantBias( B(key), B(key+1), gtsam.imuBias.ConstantBias(), \
        gtsam.noiseModel.Diagonal.Sigmas( np.sqrt(imuIntegratorOpt.deltaTij())* noiseModelBetweenBias ))
    graphFactors.add(imu_factor)
    graphFactors.add(bias_factor)
    odometry = graphValues.atPose3(X(key)).inverse()*graphValues.atPose3(X(key+1))
    #graphFactors.add(gtsam.BetweenFactorPose3(X(key), X(key+1), odometry, priorPoseNoise))


optimizer.update(graphFactors, graphValues)
optimizer.update()
#graphFactors.resize(0)
#graphValues.clear()
result = optimizer.calculateEstimate()

for i in range(n-1):
    print(result.atPose3(X(i)))
    plot_pose3("1",logSE3(result.atPose3(X(i)).matrix()),0.1)
set_axes_equal("1")

bias=[]
for i in range(n-1):
    ass = result.atConstantBias(B(i)).accelerometer()
    bias.append(ass)
bias = np.array(bias)

#plt.plot(bias[:,0],color='r', label='bias x')
#plt.plot(bias[:,1],color='g', label='bias y')
#plt.plot(bias[:,2],color='b', label='bias z')
#
plt.show()