#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transfrom_velocity import transformVel
from transfrom_imu import State, InputIMU, transformIMU
from utilities.math_tools import *
from utilities.plot_tools import *
from utilities.plot_tools import draw_cube


if __name__ == '__main__':
    figname = "test"
    fig = plt.figure(figname)
    axes = fig.add_subplot(projection='3d')

    imua = InputIMU(np.array([0, 2., 0]), np.array([0, 0, 3.5]))
    imub = InputIMU(np.array([0, 2., 0]), np.array([0, 0, 3.5]))
    Tab = p2m(np.array([0.2, 0.2, 0.1, -0.5, np.pi/2, 0.6]))
    Tba = np.linalg.inv(Tab)

    # the initial state of A
    state_a = State()
    state_a.vel = np.array([1., 0., 0.2])
    state_b = State()

    # the initial state of B
    state_b.vel = Tab[0:3, 0:3] @ transformVel(Tba, state_a.vel, state_a.omg)
    state_b.pos = Tab[0:3, 3]
    state_b.rot = Tab[0:3, 0:3]
    dt = 0.01

    cube_p = np.array([0, 0, 0])
    cube_size = np.array([0.2, 0.2, 0.1])
    A_pose_gt = np.empty((0, 3))
    B_pose_gt = np.empty((0, 3))
    B_pose_imu = np.empty((0, 3))

    for i in range(100):
        axes.cla()
        axes.set_xlim3d([-0.2, 0.5])
        axes.set_ylim3d([-0.2, 0.5])
        axes.set_zlim3d([-0.2, 0.5])

        plot_pose3(figname, state_a.pose_matrix(), 0.04, np.eye(3) * 0.001)
        plot_pose3(figname, state_b.pose_matrix(), 0.04, np.eye(3) * 0.001)
        draw_cube(axes, makeT(state_a.rot, state_a.pos), cube_p, cube_size, color='g', alpha=0.3)

        # For a better demo, we keep changing the direction of the acc_a
        imua.acc = np.linalg.inv(state_a.rot) @ skew(imua.omg) @ state_a.vel

        # Angular acceleration
        domg = (imua.omg - state_a.omg) / dt

        # Do IMU transformation
        imub = transformIMU(Tba, imua, domg)

        # Update A and B separately by IMU.
        state_a = state_a + (imua.kinematic_model(state_a, dt))
        state_b = state_b + (imub.kinematic_model(state_b, dt))

        # Get the pose of A by IMU (ground truth)
        Twa = makeT(state_a.rot, state_a.pos)
        A_pose_gt = np.vstack((A_pose_gt, state_a.pos))
        Twa = makeT(state_a.rot, state_a.pos)
        # Calculate the ground truth pose of B
        Twb_gt = Twa @ Tab
        B_pose_gt = np.vstack((B_pose_gt, Twb_gt[0:3, 3]))
        # Get the pose of B by IMU
        B_pose_imu = np.vstack((B_pose_imu, state_b.pos))
        axes.scatter(A_pose_gt[:, 0], A_pose_gt[:, 1], A_pose_gt[:, 2], color='r', label='A ground truth', s=3)
        axes.scatter(B_pose_gt[:, 0], B_pose_gt[:, 1], B_pose_gt[:, 2], color='b', label='B ground truth', s=3)
        axes.scatter(B_pose_imu[:, 0], B_pose_imu[:, 1], B_pose_imu[:, 2], color='c', label='B by IMU', s=3)
        axes.legend()
        plt.pause(0.1)
    plt.show()
