import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from graph_optimization.graph_solver import *


"""
Use SE(3) to represent the pose of the camera.
"""


class CameraVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 6)

    def update(self, dx):
        self.x = self.x @ expSE3(dx)


class PointVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 3)

    def update(self, dx):
        self.x = self.x + dx


class ReprojEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(2), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        Twc = vertices[self.link[0]].x
        pw = vertices[self.link[1]].x
        u, K = self.z
        pc, dpcdTwc, dpcdpw = transform_inv(Twc, pw, True)
        u_reproj, dudpc = reproject(pc, K, True)
        JTwc = dudpc @ dpcdTwc
        Jpw = dudpc @ dpcdpw
        return u_reproj-u, [JTwc, Jpw]


class CamerabetweenEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        Ti = vertices[self.link[0]].x
        Tj = vertices[self.link[1]].x
        Tij = np.linalg.inv(Ti) @ Tj

        r = expSE3(np.linalg.inv(self.z) @ Tij)

        Tji = np.linalg.inv(Tij)
        Rji, tji = makeRt(Tji)
        J = np.zeros([6, 6])
        J[0:3, 0:3] = -Rji
        J[3:6, 3:6] = -Rji
        J[0:3, 3:6] = Rji @ skew(tji)
        return r, [J, np.eye(6)]


class CameraEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        T = np.linalg.inv(self.z) @ vertices[self.link[0]].x
        r = logSE3(T)
        return r, [np.eye(6)]


class PointEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(3), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        r = vertices[self.link[0]].x - self.z
        return r, [np.eye(3)]


def transform(x, p, calcJ=False):
    if x.shape[0] == 6:
        T = expSE3(x)
    else:
        T = x
    R, t = makeRt(T)
    r = R @ p + t
    if (calcJ is True):
        R, t = makeRt(T)
        M = R @ skew(-p)
        dTdx = np.hstack([R, M])
        dTdp = R
        return r, dTdx, dTdp
    else:
        return r


def transform_inv(x, p, calcJ=False):
    if x.shape[0] == 6:
        T = expSE3(x)
    else:
        T = x
    Tinv = np.linalg.inv(T)
    Rinv, tinv = makeRt(Tinv)
    r = Rinv @ p + tinv
    if (calcJ is True):
        M1 = -np.eye(3)
        M2 = skew(r)
        dTdx = np.hstack([M1, M2])
        dTdp = Rinv
        return r, dTdx, dTdp
    else:
        return r


def reproject(pc, K, calcJ=False):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x, y, z = pc
    z_2 = z * z
    r = np.array([(x * fx / z + cx),
                  (y * fy / z + cy)])
    if (calcJ is True):
        J = np.array([[fx / z,    0, -fx * x / z_2],
                      [0, fy / z, -fy * y / z_2]])
        return r, J
    else:
        return r


def reproj_error0(x_cw, pw, u, K, calcJ=False):
    """
    reproject a world point into camera coordinates by Tcw.
    r = reproject(transform(x_cw, pw), K) - u
    """
    if (calcJ is True):
        pc, dpcdTwc, dpcdpw = transform(x_cw, pw, True)
        u_proj, dudpc = reproject(pc, K, True)
        dudTwc = dudpc @ dpcdTwc
        dudpw = dudpc @ dpcdpw
        return u_proj-u, dudTwc, dudpw
    else:
        pc = transform(x_cw, pw)
        u_proj = reproject(pc, K)
        return u_proj-u


def reproj_error(x_wc, pw, u, K, calcJ=False):
    """
    reproject a world point into camera coordinates by Twc.
    """
    if (calcJ is True):
        pc, dpcdTwc, dpcdpw = transform_inv(x_wc, pw, True)
        u_proj, dudpc = reproject(pc, K, True)
        dudTwb = dudpc @ dpcdTwc
        dudpw = dudpc @ dpcdpw
        return u_proj-u, dudTwb, dudpw
    else:
        pc = transform_inv(x_wc, pw)
        u_proj = reproject(pc, K)
        return u_proj-u


def reproj_error_with_bc(x_wc, pw, u, K, x_bc=np.zeros(6), calcJ=False):
    """
    reproject a world point into camera coordinates by Twb and Tbc.
    """
    if (calcJ is True):
        x_wc, dTwcdTwb, _ = pose_plus(x_wb, x_bc, True)
        pc, dpcdTwc, dpcdpw = transform_inv(x_wc, pw, True)
        u_proj, dudpc = reproject(pc, K, True)
        dudTwb = dudpc @ dpcdTwc @ dTwcdTwb
        dudpw = dudpc @ dpcdpw
        return u_proj-u, dudTwb, dudpw
    else:
        x_wc = pose_plus(x_wb, x_bc)
        pc = transform_inv(x_wc, pw)
        u_proj = reproject(pc, K)
        return u_proj-u


def pose_plus(x1, x2, calcJ=False):
    T1 = expSE3(x1)
    T2 = expSE3(x2)
    x3 = logSE3(T1 @ T2)
    if (calcJ is True):
        J1 = np.zeros([6, 6])
        R1, t1 = makeRt(T1)
        R2, t2 = makeRt(T2)
        J1[0:3, 0:3] = R2.T
        J1[3:6, 3:6] = R2.T
        J1[0:3, 3:6] = R2.T @ skew(-t2)
        J2 = np.eye(6)
        return x3, J1, J2
    else:
        return x3


def pose_minus(x1, x2, calcJ=False):
    T1 = expSE3(x1)
    T2 = expSE3(x2)
    T = np.linalg.inv(T2) @ T1
    r = logSE3(T)
    if (calcJ is True):
        R, t = makeRt(T)
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
        T = expSE3(x)
    else:
        T = x
    Tinv = np.linalg.inv(T)
    xinv = logSE3(Tinv)
    if (calcJ is True):
        R, t = makeRt(T)
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
    x3m = logSE3(expSE3(x1) @ (expSE3(x2)))
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

    print('test reproject error')
    r, J = reproject(p, K, True)
    Jm = numericalDerivative(reproject, [p, K], 0)
    check(Jm, J)

    print('test reproj_error error')
    pim = np.array([50., 60.])
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    p = np.array([5., 6., 10.])
    r, J1, J2 = reproj_error(x, p, pim, K, True)
    J1m = numericalDerivative(reproj_error, [x, p, pim, K], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj_error, [x, p, pim, K], 1)
    check(J1m, J1)
    check(J2m, J2)

    print('test reproj_error0 error')
    pim = np.array([50., 60.])
    x = np.array([0.1, 0.3, 0.5, 0.1, 0.2, 0.3])
    p = np.array([5., 6., 10.])
    r, J1, J2 = reproj_error0(x, p, pim, K, True)
    J1m = numericalDerivative(reproj_error0, [x, p, pim, K], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj_error0, [x, p, pim, K], 1)
    check(J1m, J1)
    check(J2m, J2)
