import numpy as np
from scipy.spatial.transform import Rotation

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



def transform2d(x,p):
    t = x[0:2]
    R = np.array([[np.cos(x[2]),-np.sin(x[2])], [np.sin(x[2]),np.cos(x[2])]])
    element = int(p.size/2)
    tp = np.dot(R,p).reshape(2, -1) + np.array([t,]*(element)).transpose()
    return tp

def transform3d(x,p):
    t = x[0:3]
    R = expSO3(x[3:6])
    element = int(p.size/3)
    tp = np.dot(R,p).reshape(3, -1) + np.array([t,]*(element)).transpose()
    return tp

# 3d Rotation Matrix to so3
def logSO3(mat):
    rot = Rotation.from_matrix(mat)
    q = rot.as_quat()
    squared_n = np.dot(q[0:3], q[0:3])
    if (squared_n < epsilon * epsilon):
        squared_w = q[3] * q[3]
        two_atan_nbyw_by_n = 2. / q[3] - (2.0/3.0) * (squared_n) / (q[3] * squared_w)
    else:
        n = np.sqrt(squared_n)
        if (np.abs(q[3]) < epsilon):
            if (q[3] > 0.):
              two_atan_nbyw_by_n = np.pi / n
            else:
              two_atan_nbyw_by_n = -np.pi / n
        else:
            two_atan_nbyw_by_n = 2. * np.arctan(n / q[3]) / n
    return two_atan_nbyw_by_n * q[0:3]
    
# so3 to 3d Rotation Matrix
def expSO3(v):
    theta_sq = np.dot(v, v)
    imag_factor = 0.
    real_factor = 0.
    if (theta_sq < epsilon * epsilon):
        theta_po4 = theta_sq * theta_sq
        imag_factor = 0.5 - (1.0 / 48.0) * theta_sq + (1.0 / 3840.0) * theta_po4
        real_factor = 1. - (1.0 / 8.0) * theta_sq +   (1.0 / 384.0) * theta_po4
    else:
        theta = np.sqrt(theta_sq)
        half_theta = 0.5 * theta
        sin_half_theta = np.sin(half_theta)
        imag_factor = sin_half_theta / theta
        real_factor = np.cos(half_theta)
    quat = np.array([imag_factor*v[0], imag_factor*v[1], imag_factor*v[2], real_factor])
    rot = Rotation.from_quat(quat)
    return rot.as_matrix()



if __name__ == '__main__':
    quat = np.array([0.        , 0.67385289, 0.44923526, 0.58660887])
    rot = Rotation.from_quat(quat)
    v =  logSO3(rot.as_matrix())
    q2 =  expSO3(v)
    print(q2)