#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


class InputIMU:
    def __init__(self, acc, omg):
        self.acc = acc
        self.omg = omg

    def kinematic_model(self, state, dt):
        """
        kinematic model for imu input
        pos = pos_old + vel * dt
        rot = imu.omg - bias_omg
        """
        f = np.zeros(12, )
        f[0:3] = state.vel * dt
        f[3:6] = self.omg * dt
        f[6:9] = state.rot @ self.acc * dt
        f[9:12] = self.omg - state.omg
        return f


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

    def pose_matrix(self):
        m = np.eye(4)
        m[0:3, 0:3] = self.rot
        m[0:3, 3] = self.pos
        return m


if __name__ == '__main__':
    pass
