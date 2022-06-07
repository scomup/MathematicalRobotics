
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

prevVel = np.array([0,0,0])
prevState = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , np.array([0,0,0]))

imu_params = gtsam.PreintegrationParams.MakeSharedU(9.8)
imu_params.setAccelerometerCovariance(np.eye(3) * np.power( 1e-18, 2))
imu_params.setIntegrationCovariance(np.eye(3) * np.power(0, 2))
imu_params.setGyroscopeCovariance(np.eye(3) * np.power(0.00175, 2))
imuIntegrator = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias())


startState = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , gtsam.Point3(0,0,0))
currState = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0)) , gtsam.Point3(0,0,0))
trj = []
for i in range(100):
    imuIntegrator.integrateMeasurement(np.array([0.5,0,9.8]), np.array([0.0,0,0.2]), 0.01)
    currState = imuIntegrator.predict(startState, gtsam.imuBias.ConstantBias())
    trj.append([currState.pose().translation()[0],currState.pose().translation()[1],currState.pose().translation()[2]])
imu_factor = gtsam.ImuFactor(0, 1, 2, 3, 4, imuIntegrator)
pi = gtsam.Pose3(gtsam.Rot3.Quaternion(1,0,0,0), gtsam.Point3(0,0,0))
pj = gtsam.Pose3(gtsam.Rot3.Quaternion(0.707,0,0,0.707), gtsam.Point3(0,0,0))
vi = np.array([0,0,0.])
vj = np.array([0,0,0.])
ni = gtsam.NavState(pi,vi)
nj = gtsam.NavState(pj,vj)
nj_predict = currState
res = nj.localCoordinates(nj_predict)

dR = nj.pose().rotation().inverse()*nj_predict.pose().rotation()
dr = logSO3(dR.matrix())
dt = nj.pose().rotation().inverse().matrix().dot((nj_predict.pose().translation() - nj.pose().translation()))
dv = nj.pose().rotation().inverse().matrix().dot((nj_predict.velocity() - nj.velocity()))
"""
  const Rot3 dR = R_.between(g.R_, H1 ? &D_dR_R : 0);
  const Point3 dP = R_.unrotate(g.t_ - t_, H1 ? &D_dt_R : 0);
  const Vector dV = R_.unrotate(g.v_ - v_, H1 ? &D_dv_R : 0);

  Vector9 xi;
  Matrix3 D_xi_R;
  xi << Rot3::Logmap(dR), dP, dV;

  return xi;
"""
bias = gtsam.imuBias.ConstantBias()
e = imu_factor.evaluateError(pi,vi,pj,vj,bias)
e2 = e.dot(e)
print(e2)
pj = currState.pose()
vj = currState.velocity()
e = imu_factor.evaluateError(pi,vi,pj,vj,bias)
e2 = e.dot(e)
print(e2)

# The following argument types are supported:
#    1. (self: gtsam.gtsam.ImuFactor, 
# pose_i: gtsam.gtsam.Pose3, 
# vel_i: numpy.ndarray[numpy.float64[m, 1]], 
# pose_j: gtsam.gtsam.Pose3, 
# vel_j: numpy.ndarray[numpy.float64[m, 1]], 
# bias: gtsam.gtsam.imuBias.ConstantBias) -> numpy.ndarray[numpy.float64[m, 1]]
trj = np.array(trj)
plt.plot(trj[:,0],trj[:,1],label='trj_ref')
plt.legend()
plt.show()
