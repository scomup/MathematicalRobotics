import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def transform(x, p, calcJ = False):
    R = expSO3(x[0:3])
    t = x[3:6]
    r = R.dot(p) + t
    if(calcJ == True):
        M = R.dot(skew(-p))
        dTdx = np.hstack([M, R])
        dTdp = R
        return  r, dTdx, dTdp
    else:
        return r

def transformInv(x, p, calcJ = False):
    Rinv = expSO3(-x[0:3])
    t = x[3:6]
    r = Rinv.dot(p-t)
    if(calcJ == True):
        M0 = skew(Rinv.dot(p-t))
        M1 = -np.eye(3)
        dTdx = np.hstack([M0, M1])
        dTdp = Rinv
        return  r, dTdx, dTdp
    else:
        return r


def projection(pc, K, calcJ = False):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    r = np.array([(pc[0]/pc[2] * fx + cx),
                  (pc[1]/pc[2] * fy + cy)])
    if(calcJ == True):
        J = np.array([[fx/pc[2],    0, -fx*pc[0]/pc[2]**2],
                      [   0, fy/pc[2], -fy*pc[1]/pc[2]**2]])
        return  r, J
    else:
        return r

def reporj_bk(x_cw, pw, pim, K, calcJ = False):
    if(calcJ == True):
        pc, dTdx, dTdp = transform(x_cw, pw, True)
        r ,dKdT = projection(pc, K, True)
        dKdx = dKdT.dot(dTdx)
        dKdp = dKdT.dot(dTdp)
        return r-pim, dKdx, dKdp
    else:
        pc = transform(x_cw, pw)
        r = projection(pc, K)
        return r-pim

def reporj(x_wc, pw, pim, K, calcJ = False):
    if(calcJ == True):
        pc, dTdx, dTdp = transformInv(x_wc, pw, True)
        r ,dKdT = projection(pc, K, True)
        dKdx = dKdT.dot(dTdx)
        dKdp = dKdT.dot(dTdp)
        return r-pim, dKdx, dKdp
    else:
        pc = transformInv(x_wc, pw)
        r = projection(pc, K)
        return r-pim


def pose_plus(x1,x2):
    R1 = expSO3(x1[0:3])
    t1 = x1[3:6]
    R2 = expSO3(x2[0:3])
    t2 = x2[3:6]
    R = R1.dot(R2)
    t = R1.dot(t2) + t1
    return np.hstack([logSO3(R),t])

def pose_inv(x):
    R = expSO3(x[0:3])
    t = x[3:6]
    Rinv = np.linalg.inv(R)
    return np.hstack([logSO3(Rinv),Rinv.dot(-t)])

def tom(x):    
    R = expSO3(x[0:3])
    t = x[3:6]
    return makeT(R,t)

def tox(m):    
    R,t = makeRt(m)
    return np.hstack([logSO3(R),t])

if __name__ == '__main__':
    fx = 400.
    fy = 400.
    cx = 200.
    cy = 100.
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])

    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    p = np.array([5.,6.,10.])
    pim = np.array([50.,60.])
    r,J1,J2 = transform(x, p, True)
    J1m = numericalDerivative(transform, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transform, [x, p], 1)
    print('test transform error')
    if(np.linalg.norm(J1m - J1) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(J2m - J2) < 0.0001):
        print('OK')
    else:
        print('NG')

    r,J1,J2 = transformInv(x, p, True)
    J1m = numericalDerivative(transformInv, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transformInv, [x, p], 1)
    print('test transformInv error')
    if(np.linalg.norm(J1m - J1) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(J2m - J2) < 0.0001):
        print('OK')
    else:
        print('NG')

    r, J = projection(p,K, True)
    Jm = numericalDerivative(projection,[p, K], 0)
    print('test projection error')
    if(np.linalg.norm(Jm - J) < 0.0001):
        print('OK')
    else:
        print('NG')

    r,J1,J2 = reporj(x, p, pim, K, True)
    J1m = numericalDerivative(reporj, [x, p, pim, K], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reporj, [x, p, pim, K], 1)
    print('test reprojection error')
    if(np.linalg.norm(J1m - J1) < 0.0001):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(J2m - J2) < 0.0001):
        print('OK')
    else:
        print('NG')
