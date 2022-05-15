import numpy as np

epsilon = 1e-5

def v2m(v):
    return np.array([[np.cos(v[2]),-np.sin(v[2]), v[0]],
            [np.sin(v[2]),np.cos(v[2]), v[1]], 
            [0,0,1]])

def m2v(m):
    return np.array([m[0,2],m[1,2],np.arctan2(m[1,0],m[0,0])])

def p2m(x):
    t = x[0:3]
    R = expSO3(x[3:6])
    m = np.eye(4)
    m[0:3,0:3] = R
    m[0:3,3] = t
    return m

def m2p(m):
    x = np.zeros(6)
    x[0:3] = m[0:3,3]
    x[3:6] = logSO3(m[0:3,0:3])
    return x


def skew(vector):
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

def makeT(R,t):
    n = t.shape[0]
    T = np.eye(n+1)
    T[0:n,0:n] = R
    T[0:n,n] = t
    return T 

def makeRt(T):
    n = T.shape[0] - 1
    return T[0:n,0:n], T[0:n,n]

def expSE3(x):
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

def logSE3(pose):
    w = logSO3(pose[0:3,0:3])
    T = pose[0:3,3]
    t = np.linalg.norm(w)
    if (t < 1e-10):
      return np.hstack([T,w])
    else:
        W = skew(w / t)
        # Formula from Agrawal06iros, equation (14)
        # simplified with Mathematica, and multiplying in T to avoid matrix math
        Tan = np.tan(0.5 * t)
        WT = W.dot(T)
        u = T - (0.5 * t) * WT + (1 - t / (2. * Tan)) * (W.dot(WT))
        #Vector6 log
        return np.hstack([u,w])

def expSO3(omega):
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
        R = np.eye(3) + sin_theta * K + one_minus_cos * KK
        return R 

def logSO3(R):
    R11, R12, R13 = R[0, :]
    R21, R22, R23 = R[1, :]
    R31, R32, R33 = R[2, :]
    tr = np.trace(R)
    omega = np.zeros(3)
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
            #R11 is the largest diagonal, a=1, b=2, c=3
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
            #when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            #use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            #see https://github.com/borglab/gtsam/issues/746 for details
            magnitude = 0.5 - tr_3 / 12.0 + tr_3*tr_3/60.0
    omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])
    return omega

def transform2d(x,p):
    t = x[0:2]
    R = np.array([[np.cos(x[2]),-np.sin(x[2])], [np.sin(x[2]),np.cos(x[2])]])
    element = int(p.size/2)
    tp = np.dot(R,p).reshape(2, -1) + np.array([t,]*(element)).transpose()
    return tp

def transform3d(x,p, x2T = expSE3):
    R, t = makeRt(x2T(x))
    element = int(p.size/3)
    tp = np.dot(R,p).reshape(3, -1) + np.array([t,]*(element)).transpose()
    return tp

if __name__ == '__main__':
    print('test SO3')
    v = np.array([1,0.3,2])
    R = expSO3(v)
    R2 = expSO3(logSO3(R))
    if(np.linalg.norm(R - R2) < 0.0001):
        print('OK')
    else:
        print('NG')


    print('test SE3')
    v = np.array([1,0.3,2,1,-3.2,0.2])
    R = expSE3(v)
    R2 = expSE3(logSE3(R))

    if(np.linalg.norm(R - R2) < 0.0001):
        print('OK')
    else:
        print('NG')