#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.plot_pose import *
from transfrom_velocity import transformVelocity3D
from robot_geometry.demo_plane_cross_cube import draw_cube


L_OMG = 9


class InputIMU:
    def __init__(self, acc, omg):
        self.acc = acc
        self.omg = omg


def transformVel(Tba, v, omega):
    Rba, tba = makeRt(Tba)
    return Rba @ v + skew(tba) @ Rba @ omega


def transformIMU(Tba, imua, domg):
    imub = InputIMU(np.zeros(3), np.zeros(3))
    Rba, tba = makeRt(Tba)
    imub.acc = Rba @ imua.acc - skew(Rba @ imua.omg) @ skew(Rba @ imua.omg) @ tba + skew(tba) @ Rba @ domg
    imub.omg = Rba @ imua.omg
    return imub


class State:
    def __init__(self, pos=np.zeros(3, ), rot=np.eye(3), vel=np.zeros(3,), omg=np.zeros(3,)):
        self.pos = pos
        self.rot = rot
        self.vel = vel
        self.omg = omg

    @classmethod
    def form_vec(cls, vec):
        pos = vec[0:0+3]
        rot = expSO3(vec[3:6])
        vel = vec[6:9]
        omg = vec[9:12]
        return State(pos, rot, vel, omg)

    def __add__(self, f):
        r = State()
        r.pos = self.pos + f[0:3]
        r.rot = self.rot @ expSO3(f[3:6])
        r.vel = self.vel + f[6:9]
        r.omg = self.omg + f[9:12]
        return r

    def pose(self):
        m = np.eye(4)
        m[0:3, 0:3] = self.rot
        m[0:3, 3] = self.pos
        return m


def model_imu(state, u, dt):
    """
    kinematic model for imu input
    pos = pos_old + vel * dt
    rot = imu.omg - bias_omg
    """
    f = np.zeros(12, )
    f[0:3] = state.vel * dt
    f[3:6] = u.omg * dt
    f[6:9] = state.rot @ u.acc * dt
    f[9:12] = u.omg - state.omg
    return f

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

        plot_pose3(figname, state_a.pose(), 0.04, np.eye(3) * 0.001)
        plot_pose3(figname, state_b.pose(), 0.04, np.eye(3) * 0.001)
        draw_cube(axes, makeT(state_a.rot, state_a.pos), cube_p, cube_size, color='g', alpha=0.3)

        # We keep changing the direction of the acc_a
        # make the domgdt is not zero.
        imua.acc = np.linalg.inv(state_a.rot) @ skew(imua.omg) @ state_a.vel
        domg = (imua.omg - state_a.omg) / dt

        # do IMU transformation
        imub = transformIMU(Tba, imua, domg)

        # update A and B separately by IMU.
        state_a = state_a + (model_imu(state_a, imua, dt))
        state_b = state_b + (model_imu(state_b, imub, dt))

        # draw ground truth and predict pose by imu
        Twa = makeT(state_a.rot, state_a.pos)
        A_pose_gt = np.vstack((A_pose_gt, state_a.pos))
        Twa = makeT(state_a.rot, state_a.pos)
        Twb_gt = Twa @ Tab
        B_pose_gt = np.vstack((B_pose_gt, Twb_gt[0:3, 3]))
        B_pose_imu = np.vstack((B_pose_imu, state_b.pos))
        axes.scatter(A_pose_gt[:, 0], A_pose_gt[:, 1], A_pose_gt[:, 2], color='r', label='A ground truth', s=3)
        axes.scatter(B_pose_gt[:, 0], B_pose_gt[:, 1], B_pose_gt[:, 2], color='b', label='B ground truth', s=3)
        axes.scatter(B_pose_imu[:, 0], B_pose_imu[:, 1], B_pose_imu[:, 2], color='c', label='B by IMU', s=3)
        axes.legend()
        plt.pause(0.1)
    plt.show()

