import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def transform(x, p, calcJ=False):
    if x.shape[0] == 6:
        t = x[0:3]
        R = expSO3(x[3:6])
    else:
        R, t = makeRt(x)
    r = R @ p + t
    if (calcJ is True):
        M = R @ skew(-p)
        dTdx = np.hstack([R, M])
        dTdp = R
        return r, dTdx, dTdp
    else:
        return r


def transform_inv(x, p, calcJ=False):
    if x.shape[0] == 6:
        t = x[0:3]
        Rinv = expSO3(-x[3:6])
    else:
        R, t = makeRt(x)
        Rinv = np.linalg.inv(R)
    r = Rinv @ (p - t)
    if (calcJ is True):
        M1 = -np.eye(3)
        M2 = skew(Rinv @ (p-t))
        dTdx = np.hstack([M1, M2])
        dTdp = Rinv
        return r, dTdx, dTdp
    else:
        return r


def projection(pc, K, calcJ=False):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    r = np.array([(pc[0]/pc[2] * fx + cx),
                  (pc[1]/pc[2] * fy + cy)])
    if (calcJ is True):
        J = np.array([[fx/pc[2],    0, -fx*pc[0]/pc[2]**2],
                      [0, fy/pc[2], -fy*pc[1]/pc[2]**2]])
        return r, J
    else:
        return r


def reproj0(x_cw, pw, u, K, calcJ=False):
    """
    reporject a wolrd point to camera frame.
    r = projection(transform(x_cw, pw), K) - u
    """
    if (calcJ is True):
        pc, dpcdTwc, dpcdpw = transform(x_cw, pw, True)
        u_reproj, dudpc = projection(pc, K, True)
        dudTwc = dudpc @ dpcdTwc
        dudpw = dudpc @ dpcdpw
        return u_reproj-u, dudTwc, dudpw
    else:
        pc = transform(x_cw, pw)
        u_reproj = projection(pc, K)
        return u_reproj-u


def reproj(x_wb, pw, u, K, x_bc=np.zeros(6), calcJ=False):
    """
    reporject a wolrd point to camera frame.
    """
    if (calcJ is True):
        x_wc, dTwcdTwb, _ = pose_plus(x_wb, x_bc, True)
        pc, dpcdTwc, dpcdpw = transform_inv(x_wc, pw, True)
        u_reproj, dudpc = projection(pc, K, True)
        dudTwb = dudpc @ dpcdTwc @ dTwcdTwb
        dudpw = dudpc @ dpcdpw
        return u_reproj-u, dudTwb, dudpw
    else:
        x_wc = pose_plus(x_wb, x_bc)
        pc = transform_inv(x_wc, pw)
        u_reproj = projection(pc, K)
        return u_reproj-u


def pose_plus(x1, x2, calcJ=False):
    t1 = x1[0:3]
    R1 = expSO3(x1[3:6])
    t2 = x2[0:3]
    R2 = expSO3(x2[3:6])
    R = R1 @ R2
    t = R1 @ t2 + t1
    x3 = np.hstack([t, logSO3(R)])
    if (calcJ is True):
        J1 = np.zeros([6, 6])
        J1[0:3, 0:3] = R2.T
        J1[3:6, 3:6] = R2.T
        J1[0:3, 3:6] = R2.T @ skew(-t2)
        J2 = np.eye(6)
        return x3, J1, J2
    else:
        return x3


def pose_minus(x1, x2, calcJ=False):
    t1 = x1[0:3]
    R1 = expSO3(x1[3:6])
    t2 = x2[0:3]
    R2 = expSO3(x2[3:6])
    R = R2.T @ R1
    t = R2.T @ (t1 - t2)
    r = pose_plus(pose_inv(x2), x1)
    if (calcJ is True):
        Jx1 = np.eye(6)
        Jx2 = np.zeros([6, 6])
        Jx2[0:3, 0:3] = -R.T
        Jx2[3:6, 3:6] = -R.T
        Jx2[0:3, 3:6] = R.T @ skew(t)
        return r, Jx1, Jx2
    else:
        return r


def pose_inv(x, calcJ=False):
    if x.shape[0] == 6:
        t = x[0:3]
        R = expSO3(x[3:6])
    else:
        R, t = makeRt(x)
    Rinv = np.linalg.inv(R)
    xinv = np.hstack([Rinv @ (-t), logSO3(Rinv)])
    if (calcJ is True):
        J1 = np.zeros([6, 6])
        J1[0:3, 0:3] = -R
        J1[3:6, 3:6] = -R
        J1[0:3, 3:6] = R @ skew(R.T @ (-t))
        return xinv, J1
    else:
        return xinv


if __name__ == '__main__':
    fx = 400.
    fy = 400.
    cx = 200.
    cy = 100.
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    print('test pose_inv')
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    r, J = pose_inv(x, True)
    Jm = numericalDerivative(pose_inv, [x], 0, pose_plus, pose_minus)
    check(J, Jm)
    print('test pose_plus and pose_minus')
    x1 = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    x2 = np.array([0.2, 0.1, -0.2, -0.1, -0.2, -0.1])
    x3m = m2p(p2m(x1) @ (p2m(x2)))
    x3, J1, J2 = pose_plus(x1, x2, True)
    x2m = pose_minus(x3, x1)
    check(x3m, x3)
    check(x2m, x2)
    print('test pose_plus error')
    J1m = numericalDerivative(pose_plus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_plus, [x1, x2], 1, pose_plus, pose_minus)
    check(J1m, J1)
    check(J2m, J2)
    print('test pose_minus error')
    r, J1, J2 = pose_minus(x1, x2, True)
    J1m = numericalDerivative(pose_minus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_minus, [x1, x2], 1, pose_plus, pose_minus)
    check(J1m, J1)
    check(J2m, J2)

    print('test transform error')
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    p = np.array([5., 6., 10.])
    r, J1, J2 = transform(x, p, True)
    J1m = numericalDerivative(transform, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transform, [x, p], 1)
    check(J1m, J1)
    check(J2m, J2)

    print('test transform_inv error')
    r, J1, J2 = transform_inv(x, p, True)
    J1m = numericalDerivative(transform_inv, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transform_inv, [x, p], 1)
    check(J1m, J1)
    check(J2m, J2)

    print('test projection error')
    r, J = projection(p, K, True)
    Jm = numericalDerivative(projection, [p, K], 0)
    check(Jm, J)

    print('test reproj error')
    pim = np.array([50., 60.])
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    xbc = np.array([-0.1, 0.3, -0.5, 0.1, -0.2, 0.3])
    p = np.array([5., 6., 10.])
    r, J1, J2 = reproj(x, p, pim, K, xbc, True)
    J1m = numericalDerivative(reproj, [x, p, pim, K, xbc], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj, [x, p, pim, K, xbc], 1)
    check(J1m, J1)
    check(J2m, J2)

    print('test reproj0 error')
    pim = np.array([50., 60.])
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    xbc = np.array([-0.1, 0.3, -0.5, 0.1, -0.2, 0.3])
    p = np.array([5., 6., 10.])
    r, J1, J2 = reproj0(x, p, pim, K, True)
    J1m = numericalDerivative(reproj0, [x, p, pim, K], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj0, [x, p, pim, K], 1)
    check(J1m, J1)
    check(J2m, J2)