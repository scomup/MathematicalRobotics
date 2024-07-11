#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


"""
If We know:
    Tba: The rigid transform matrix A a to B
    va: Translational and angular velocities of A in its local Frame [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    Based on the assumption that A and B are attached to the same rigid body, so I want to
    compute the speed of B in its local Frame (vb).
"""


def transformVelocity3D(Tba, va):
    Rba, tba = makeRt(Tba)
    M = np.eye(6)
    M[0:3, 0:3] = Rba
    M[0:3, 3:6] = skew(tba) @ Rba
    M[3:6, 3:6] = Rba
    vb = M @ va
    return vb


def transformVelocity2D(Tba, vwa):
    Rba, tba = makeRt(Tba)
    M = np.eye(3)
    M[0:2, 0:2] = Rba
    M[0:2, 2] = hat2d(-tba)
    vwb = M @ vwa
    return vwb


def transformVel(Tba, v, omega):
    Rba, tba = makeRt(Tba)
    return Rba @ v + skew(tba) @ Rba @ omega


expSE2(np.array([-1, 1., 0.2]))

if __name__ == '__main__':

    print('test TransformVelocity3D')
    # Translational and angular velocities of A in its local Frame [v_x, v_y, v_z, omega_x, omega_y, omega_z]
    va = np.array([1, 2, 3., 0.3, 0.5, 1.])

    # The rigid transform matrix frame A a to frame B
    Tba = expSE3(np.array([-1, 1, 2., 0.2, 0.4, 0.2]))

    # The Pose of A in world
    Pwa = np.eye(4)

    # The Pose of B in world
    Pwb = Pwa @ np.linalg.inv(Tba)

    dt = 1.

    vb = transformVelocity3D(Tba, va)
    Pwb_prime = Pwb @ expSE3(vb * dt)
    Pwa_prime = Pwb_prime @ Tba
    Pwa_prime_real = Pwa @ expSE3(va * dt)

    check(Pwa_prime_real, Pwa_prime)

    print('test transformVelocity2D')
    # Translational and angular velocities of A in its local Frame [v_x, v_y, omega_x, omega_y]
    va = np.array([1, 1, 0.3])
    # The rigid transform matrix frame A a to frame B
    Tba = expSE2(np.array([1, 2, 0.3]))

    # The Pose of A in world
    Pwa = np.eye(3)

    # The Pose of B in world
    Pwb = Pwa @ np.linalg.inv(Tba)

    dt = 1.

    # lb = np.array([0.85083178, 2.48670662, 0.3])
    # lb = logSE2(np.linalg.inv(Pwb) @ Pwa_prime_real @ np.linalg.inv(Tba))

    Pwa_prime_real = Pwa @ expSE2(va * dt)

    vb = transformVelocity2D(Tba, va)

    Pwa_prime = Pwb @ expSE2(vb * dt) @ Tba

    check(Pwa_prime_real, Pwa_prime)
