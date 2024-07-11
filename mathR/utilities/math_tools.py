import numpy as np

epsilon = 1e-5


def v2m(v):
    """
    covert a 3d pose vector(a 2d translation and a so2) to 2d transformation matrix
    """
    return np.array([[np.cos(v[2]), -np.sin(v[2]), v[0]],
                     [np.sin(v[2]), np.cos(v[2]), v[1]],
                     [0, 0, 1]])


def m2v(m):
    """
    covert a 2d transformation matrix to 3d pose vector(a 2d translation and a so2)
    """
    return np.array([m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])])


def p2m(x):
    """
    covert a 6d pose vector(a 3d translation and a so3) to 3d transformation matrix
    """
    t = x[0:3]
    R = expSO3(x[3:6])
    m = np.eye(4)
    m[0:3, 0:3] = R
    m[0:3, 3] = t
    return m


def m2p(m):
    """
    covert a 3d transformation matrix to 6d pose vector(a 3d translation and a so3)
    """
    x = np.zeros(6)
    x[0:3] = m[0:3, 3]
    x[3:6] = logSO3(m[0:3, 0:3])
    return x


def expSO2(v):
    v = np.atleast_1d(v)
    return np.array([[np.cos(v[0]), -np.sin(v[0])], [np.sin(v[0]), np.cos(v[0])]])


def logSO2(m):
    return np.arctan2(m[1, 0], m[0, 0])


def skew(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def unskew(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


def hat2d(v):
    return np.array([-(v)[1], (v)[0]])


def makeT(R, t):
    n = t.shape[0]
    T = np.eye(n+1)
    T[0:n, 0:n] = R
    T[0:n, n] = t
    return T


def makeRt(T):
    n = T.shape[0] - 1
    return T[0:n, 0:n], T[0:n, n]


def left_jacobian(phi):
    if np.isclose(phi, 0.):
        return np.identity(2) + 0.5 * wedge(phi)
    s = np.sin(phi)
    c = np.cos(phi)
    return (s / phi) * np.identity(2) + ((1 - c) / phi) * wedge(1.)


def inv_left_jacobian(phi):
    if np.isclose(phi, 0.):
        return np.identity(2) - 0.5 * wedge(phi)
    half_angle = 0.5 * phi
    cot_half_angle = 1. / np.tan(half_angle)
    return half_angle * cot_half_angle * np.identity(2) - half_angle * wedge(1.)


def wedge(phi):
    phi = np.atleast_1d(phi)
    Phi = np.zeros([len(phi), 2, 2])
    Phi[:, 0, 1] = -phi
    Phi[:, 1, 0] = phi
    return np.squeeze(Phi)


def expSE2(xi):
    rho = xi[0:2]
    phi = xi[2]
    R = expSO2(phi)
    t = left_jacobian(phi).dot(rho)
    return makeT(R, t)


def logSE2(T):
    R, t = makeRt(T)
    phi = logSO2(R)
    rho = inv_left_jacobian(phi).dot(t)
    return np.hstack([rho, phi])


def expSE3(x):
    """
    Exponential map of SE3
    The proof is shown in 3d_transformation_group.md (13)~(18)
    """
    omega = x[3:6]
    v = x[0:3]
    R = expSO3(omega)
    theta2 = omega.dot(omega)
    if theta2 > epsilon:
        t_parallel = omega * omega.dot(v)
        omega_cross_v = np.cross(omega, v)
        t = (omega_cross_v - R.dot(omega_cross_v) + t_parallel) / theta2
        return makeT(R, t)
    else:
        return makeT(R, v)


def expSE3test(x):
    hat_x = np.zeros([4, 4])
    omega = x[3:6]
    v = x[0:3]
    hat_x[0:3, 0:3] = skew(omega)
    hat_x[0:3, 3] = v
    hat_x_powk = hat_x.copy()
    T = np.eye(4)
    k_factorial = 1
    for k in range(1, 20):
        k_factorial *= k
        T += hat_x_powk / k_factorial
        hat_x_powk = hat_x_powk.dot(hat_x)
    return T


def logSE3(pose):
    """
    Logarithm map of SE3
    The proof is shown in 3d_transformation_group.md (19)~(24)
    """
    w = logSO3(pose[0:3, 0:3])
    T = pose[0:3, 3]
    t = np.linalg.norm(w)
    if (t < 1e-10):
        return np.hstack([T, w])
    else:
        W = skew(w / t)
        Tan = np.tan(0.5 * t)
        WT = W.dot(T)
        u = T - (0.5 * t) * WT + (1 - t / (2. * Tan)) * (W.dot(WT))
        # Vector6 log
        return np.hstack([u, w])


def expSO3(omega):
    """
    Exponential map of SO3
    The proof is shown in 3d_rotation_group.md (10)
    """
    theta2 = omega.dot(omega)
    theta = np.sqrt(theta2)
    nearZero = theta2 <= epsilon
    W = skew(omega)
    if (nearZero):
        return np.eye(3) + W
    else:
        K = W/theta
        KK = K.dot(K)
        sin_theta = np.sin(theta)
        one_minus_cos = 1 - np.cos(theta)
        R = np.eye(3) + sin_theta * K + one_minus_cos * KK  # rotation.md (10)
        return R


def expSO3test(x):
    hat_x = skew(x)
    T = np.eye(3)
    k_factorial = 1
    hat_x_powk = hat_x.copy()
    for k in range(1, 20):
        k_factorial *= k
        T += hat_x_powk / k_factorial
        hat_x_powk = hat_x_powk.dot(hat_x)
    return T


def logSO3(R):
    """
    Logarithm map of SO3
    The proof is shown in rotation.md (14)
    """
    R11, R12, R13 = R[0, :]
    R21, R22, R23 = R[1, :]
    R31, R32, R33 = R[2, :]
    tr = np.trace(R)
    omega = np.zeros(3)
    v = np.array([R32 - R23, R13 - R31, R21 - R12])
    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    # we do something special
    if (tr + 1.0 < 1e-3):
        if (R33 > R22 and R33 > R11):
            # R33 is the largest diagonal, a=3, b=1, c=2
            W = R21 - R12
            Q1 = 2.0 + 2.0 * R33
            Q2 = R31 + R13
            Q3 = R23 + R32
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W)
            sgn_w = np.sign(W)
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
        elif (R22 > R11):
            # R22 is the largest diagonal, a=2, b=3, c=1
            W = R13 - R31
            Q1 = 2.0 + 2.0 * R22
            Q2 = R23 + R32
            Q3 = R12 + R21
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W)
            sgn_w = np.sign(W)
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
        else:
            # R11 is the largest diagonal, a=1, b=2, c=3
            W = R32 - R23
            Q1 = 2.0 + 2.0 * R11
            Q2 = R12 + R21
            Q3 = R31 + R13
            r = np.sqrt(Q1)
            one_over_r = 1 / r
            norm = np.sqrt(Q1*Q1 + Q2*Q2 + Q3*Q3 + W*W)
            sgn_w = np.sign(W)
            mag = np.pi - (2 * sgn_w * W) / norm
            scale = 0.5 * one_over_r * mag
            omega = sgn_w * scale * np.array([Q2, Q3, Q1])
    else:
        magnitude = 0
        tr_3 = tr - 3.0
        if (tr_3 < -1e-6):
            # this is the normal case -1 < trace < 3
            theta = np.arccos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            # see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3*tr_3/60.0
        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])
    return omega


def transform2d(x, p, x2T=lambda x: x):
    R, t = makeRt(x2T(x))
    element = int(p.size/2)
    tp = np.dot(R, p).reshape(2, -1) + np.array([t, ]*(element)).transpose()
    return tp


def transform3d(x, p, x2T=lambda x: x):
    R, t = makeRt(x2T(x))
    element = int(p.size/3)
    tp = np.dot(R, p).reshape(3, -1) + np.array([t, ]*(element)).transpose()
    return tp


def numericalDerivative(func, param, idx, plus=lambda a, b: a + b, minus=lambda a, b: a - b, delta=1e-5):
    r = func(*param)
    m = r.shape[0]
    n = param[idx].shape[0]
    J = np.zeros([m, n])
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = delta
        param_delta = param.copy()
        param_delta[idx] = plus(param[idx], dx)
        J[:, j] = minus(func(*param_delta), r)/delta
    return J


def HSO3(omega):
    theta2 = omega.dot(omega)
    theta = np.sqrt(theta2)
    near_zero = theta2 <= epsilon
    W = skew(omega)
    if (near_zero):
        dexp = np.eye(3) - 0.5 * W
    else:
        K = W / theta
        KK = K.dot(K)
        sin_theta = np.sin(theta)
        s2 = np.sin(theta / 2.0)
        one_minus_cos = 2.0 * s2 * s2  # [1 - cos(theta)]
        a = one_minus_cos / theta
        b = 1.0 - sin_theta / theta
        dexp = np.eye(3) - a * K + b * KK
    return dexp


def dHinvSO3(omega, v):
    theta2 = omega.dot(omega)
    theta = np.sqrt(theta2)
    H = HSO3(omega)
    Hinv = np.linalg.inv(H)
    W = skew(omega)
    K = W / theta
    c = Hinv.dot(v)
    theta2 = omega.dot(omega)
    theta = np.sqrt(theta2)
    near_zero = theta2 <= epsilon
    if (near_zero):
        dHinv = skew(c) * 0.5
    else:
        sin_theta = np.sin(theta)
        s2 = np.sin(theta / 2.0)
        one_minus_cos = 2.0 * s2 * s2  # [1 - cos(theta)]
        Kv = K.dot(c)
        a = one_minus_cos / theta
        b = 1.0 - sin_theta / theta
        Da = (sin_theta - 2.0 * a) / theta2
        Db = (one_minus_cos - 3.0 * b) / theta2
        tmp = (Db * K - Da * np.eye(3)).dot(Kv.reshape([3, 1]).dot(omega.reshape([1, 3]))) - \
            skew(Kv * b / theta) + \
            (a * np.eye(3) - b * K).dot(skew(c / theta))
        dHinv = -Hinv.dot(tmp)
    return dHinv


def dLogSO3(omega):
        theta2 = omega.dot(omega)
        if (theta2 <= epsilon):
            return np.eye(3)
        theta = np.sqrt(theta2)
        W = skew(omega)
        WW = W.dot(W)
        return np.eye(3) + 0.5 * W + (1 / (theta * theta) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * WW


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    n = np.linalg.norm(q)
    if np.any(n == 0.0):
        raise ZeroDivisionError("bad quaternion input")
    else:
        m = np.empty((3, 3))
        m[0, 0] = 1.0 - 2*(y**2 + z**2)/n
        m[0, 1] = 2*(x*y - z*w)/n
        m[0, 2] = 2*(x*z + y*w)/n
        m[1, 0] = 2*(x*y + z*w)/n
        m[1, 1] = 1.0 - 2*(x**2 + z**2)/n
        m[1, 2] = 2*(y*z - x*w)/n
        m[2, 0] = 2*(x*z - y*w)/n
        m[2, 1] = 2*(y*z + x*w)/n
        m[2, 2] = 1.0 - 2*(x**2 + y**2)/n
        return m


def matrix_to_quaternion(matrix):
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    else:
        if matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w])


def check(a, b):
    if (np.linalg.norm(a - b) < 0.0001):
        print('OK')
    else:
        print('NG')

if __name__ == '__main__':
    print('test HSO3')
    x = np.array([0.5, 0.6, 0.7])
    dx = np.array([0.02, 0.03, 0.03])
    R1 = (expSO3(x+dx))
    R2 = (expSO3(x).dot(expSO3(HSO3(x).dot(dx))))
    check(R1, R2)

    # exit(0)
    print('test SO3')
    v = np.array([1, 0.3, 2])
    R = expSO3(v)
    R2 = expSO3(logSO3(R))
    R3 = expSO3test(logSO3(R))
    check(R, R2)
    check(R2, R3)

    print('test SE3')
    v = np.array([1, 0.3, 2, 1, -3.2, 0.2])
    R = expSE3(v)
    R2 = expSE3(logSE3(R))
    R3 = expSE3test(logSE3(R))
    check(R, R2)
    check(R2, R3)

    x = np.array([0.5, 0.2, 0.2])
    R = expSO3(x)

    print('test numerical derivative')

    def residual(x, a):
        """
        residual function of 3D rotation (SO3)
        guass_newton_method.md (7)
        """
        R = expSO3(x)
        r = R.dot(a)
        return r.flatten()

    def plus(x1, x2):
        """
        The increment of SO3
        guass_newton_method.md (5)
        """
        return logSO3(expSO3(x1).dot(expSO3(x2)))

    a = np.array([1., 2., 3.])
    """
    The jocabian of 3D rotation (SO3)
    guass_newton_method.md (9)
    """
    J = -R.dot(skew(a))
    J_numerical = numericalDerivative(residual, [x, a], 0, plus)
    check(J, J_numerical)
    x1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    x2 = np.array([-0.3, -0.4, 0.1, 0.3, 0.5, 0.7])

    def plus(x1, x2):
        return m2p(p2m(x1) @ p2m(x2))

    def minus(x1, x2):
        r_rot = logSO3(np.linalg.inv(expSO3(x2[3:])) @ expSO3(x1[3:]))
        r_trans = np.linalg.inv(expSO3(x2[3:])) @ (x1[:3] - x2[:3])
        # return np.concatenate([r_trans, r_rot])
        return np.concatenate([r_trans, r_rot])

    J_numerical = numericalDerivative(minus, [x1, x2], 1, plus, minus)
