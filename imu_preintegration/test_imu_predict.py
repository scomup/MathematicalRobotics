import preintegration
# import preintegration1
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import os
    FILE_PATH = os.path.join(os.path.dirname(__file__), '..')
    imu_file = FILE_PATH+'/data/imu_data.npy'
    truth_file = FILE_PATH+'/data/truth_pose.npy'

    imu = np.load(imu_file)
    truth = np.load(truth_file)

    import gtsam
    G = 9.8
    imu_params = gtsam.PreintegrationParams.MakeSharedU(G)
    prevBias = gtsam.imuBias.ConstantBias(np.array([0, 0, 0]), np.array([0, 0, 0]))
    imuIntegratorGT = gtsam.PreintegratedImuMeasurements(
        imu_params, gtsam.imuBias.ConstantBias(np.array([0.0, -0.0, -0.0]), np.array([0., 0., 0.])))

    imuIntegrator0 = preintegration.ImuIntegration(G)
    # imuIntegrator1 = preintegration1.ImuIntegration(G)
    lastImuTime = -1
    stateGT = gtsam.NavState(gtsam.Pose3(gtsam.Rot3.Quaternion(1, 0, 0, 0), gtsam.Point3(0, 0, 0)), np.array([0, 0, 0]))
    state0 = preintegration.NavState(np.eye(3), np.array([0, 0, 0]), np.array([0, 0, 0]))
    # state1 = preintegration1.NavState(np.eye(3), np.array([0, 0, 0]), np.array([0, 0, 0]))

    trj0 = []
    trj1 = []
    trj2 = []
    for i in imu:
        imuTime = i[0]
        dt = 0.01
        imuIntegratorGT.integrateMeasurement(i[1:4], i[4:7], dt)
        currStateGT = imuIntegratorGT.predict(stateGT,
                                              gtsam.imuBias.ConstantBias(np.array([0.01, 0.01, 0.02]),
                                                                         np.array([0.0, 0.0, 0.0])))
        imuIntegrator0.update(i[1:4], i[4:7], dt)
        # imuIntegrator1.update(i[1:4], i[4:7], dt)
        currState0 = imuIntegrator0.predict(state0, np.array([0.01, 0.01, 0.02, 0.0, 0.0, 0.0]))
        # currState1 = imuIntegrator1.predict(state1, np.array([0.01, 0.01, 0.02, 0.0, 0.0, 0.0]))
        trj0.append([currState0.p[0], currState0.p[1], currState0.p[2]])
        # trj1.append([currState1.p[0], currState1.p[1], currState1.p[2]])
        trj2.append([currStateGT.pose().translation()[0],
                     currStateGT.pose().translation()[1],
                     currStateGT.pose().translation()[2]])
        # if (imuIntegrator0.d_tij > 20):
        #    break
    trj0 = np.array(trj0)
    # trj1 = np.array(trj1)
    trj2 = np.array(trj2)
    plt.plot(trj0[:, 0], trj0[:, 1], label='trj_0')
    # plt.plot(trj1[:, 0], trj1[:, 1], label='trj_1')
    plt.plot(trj2[:, 0], trj2[:, 1], label='trj_ref')
    plt.plot(truth[:, 1], truth[:, 2], label='truth')
    plt.grid()
    plt.legend()
    plt.show()
