import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
import yaml
from imu_preintegration.preintegration import *


def readimu(n, folder):
    frames = []
    imus = np.zeros([0, 7])
    for idx in range(0, n):
        fn = folder+'/F%04d.yaml' % idx
        print('read %s...' % fn)
        with open(fn) as file:
            vertex = yaml.safe_load(file)
            imu = np.array(vertex['imu']['data']).reshape(vertex['imu']['num'], -1)
            imus = np.append(imus, imu, axis=0)
    return imus


if __name__ == '__main__':
    import os
    FILE_PATH = os.path.join(os.path.dirname(__file__), '..')
    # imu = readimu(300, 'data/slam')
    imu = np.load('/home/liu/imu.npy')
    import gtsam
    G = 9.81
    imu_params = gtsam.PreintegrationParams.MakeSharedU(G)
    prevBias = gtsam.imuBias.ConstantBias(np.array([0, 0, 0]), np.array([0, 0, 0]))
    imuIntegratorGT = gtsam.PreintegratedImuMeasurements(imu_params,
                                                         gtsam.imuBias.ConstantBias(np.array([0.0, -0.0, -0.0]),
                                                                                    np.array([0., 0., 0.])))

    imuIntegrator0 = ImuIntegration(G)
    # imuIntegrator1 = preintegration1.ImuIntegration(G)
    lastImuTime = -1
    stateGT = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1, 0, 0, 0), gtsam.Point3(0, 0, 0)), np.array([0, 0, 0]))
    state0 = NavState(np.eye(3), np.array([0, 0, 0]), np.array([0, 0, 0]))
    # state1 = preintegration1.NavState(np.eye(3), np.array([0, 0, 0]), np.array([0, 0, 0]))

    trj0 = []
    trj1 = []
    trj2 = []
    for i in imu:
        imuTime = i[0]
        dt = 1/400.
        imuIntegratorGT.integrateMeasurement(i[1:4], i[4:7], dt)
        currStateGT = imuIntegratorGT.predict(stateGT, gtsam.imuBias.ConstantBias())
        imuIntegrator0.update(i[1:4], i[4:7], dt)
        # imuIntegrator1.update(i[1:4], i[4:7], dt)
        currState0 = imuIntegrator0.predict(state0, np.zeros(6))
        # currState1 = imuIntegrator1.predict(state1, np.array([0.01, 0.01, 0.02, 0.0, 0.0, 0.0]))
        trj0.append([currState0.p[0], currState0.p[1], currState0.p[2]])
        # trj1.append([currState1.p[0], currState1.p[1], currState1.p[2]])
        trj2.append([currStateGT.pose().translation()[0],
                     currStateGT.pose().translation()[1],
                     currStateGT.pose().translation()[2]])
        if (imuIntegrator0.d_tij > 1):
            break
    trj0 = np.array(trj0)
    # trj1 = np.array(trj1)
    trj2 = np.array(trj2)
    plt.plot(trj0[:, 0], trj0[:, 1], label='trj_0')
    # plt.plot(trj1[:, 0], trj1[:, 1], label='trj_1')
    plt.plot(trj2[:, 0], trj2[:, 1], label='trj_ref')
    # plt.plot(truth[:, 1], truth[:, 2], label='truth')
    plt.grid()
    plt.legend()
    plt.show()
