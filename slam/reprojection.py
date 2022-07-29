import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def BinvAB(x_a, x_b, calcJ = False):
    A = tom(x_a)
    B = tom(x_b)
    Binv = np.linalg.inv(B)
    r = tox(Binv.dot(A.dot(B)))
    if(calcJ == True):
        Ja = np.zeros([6,6])
        Rb = B[0:3,0:3]
        Ja[0:3,0:3] = Rb.T
        Ja[3:6,3:6] = Rb.T
        Ja[3:6,0:3] = Rb.T.dot(skew(-x_b[3:6]))
        return r, Ja
    else:
        return r
    

def getTcicj(x_wbi, x_wbj, x_bc, calcJ = False):

    if(calcJ == True):
        x_bibj, Jdxj, Jdxi = pose_minus(x_wbj, x_wbi, True)
        x_cicj, Jh = BinvAB(x_bibj, x_bc, True)
        numericalDerivative(getTcicj, [xi,xj,xbc], 0, pose_plus, pose_minus)
        return r, Jh.dot(Jdxi), Jh.dot(Jdxj)
    else:
        x_bibj = pose_minus(x_wbj, x_wbi)
        x_cicj = BinvAB(x_bibj, x_bc)
        return x_cicj


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

def reporj(x_wb, pw, u, K, x_bc = np.zeros(6), calcJ = False):
    if(calcJ == True):
        x_wc, dTwcdTwb, _ = pose_plus(x_wb, x_bc, True)
        pc, dpcdTwc, dpcdpw = transformInv(x_wc, pw, True)
        u_reproj ,dudpc = projection(pc, K, True)
        dudTwb = dudpc.dot(dpcdTwc.dot(dTwcdTwb))
        dudpw = dudpc.dot(dpcdpw)
        return u_reproj-u, dudTwb, dudpw
    else:
        x_wc = pose_plus(x_wb, x_bc)
        pc = transformInv(x_wc, pw)
        u_reproj = projection(pc, K)
        return u_reproj-u


def pose_plus(x1,x2, calcJ = False):
    R1 = expSO3(x1[0:3])
    t1 = x1[3:6]
    R2 = expSO3(x2[0:3])
    t2 = x2[3:6]
    R = R1.dot(R2)
    t = R1.dot(t2) + t1
    x3 = np.hstack([logSO3(R),t])
    if(calcJ == True):
        J1 = np.eye(6)
        J1[0:3,0:3] = R2.T
        J1[3:6,3:6] = R2.T
        J1[3:6,0:3] = R2.T.dot(skew(-t2))
        J2 = np.eye(6)
        return x3, J1, J2
    else:
        return x3

def pose_minus(x1,x2,calcJ = False):
    R1 = expSO3(x1[0:3])
    t1 = x1[3:6]
    R2 = expSO3(x2[0:3])
    t2 = x2[3:6]
    R = R2.T.dot(R1)
    t = R2.T.dot(t1-t2)
    r = pose_plus(pose_inv(x2),x1)
    if(calcJ == True):
        Jx1 = np.eye(6)
        Jx2 = np.zeros([6,6])
        Jx2[0:3,0:3] = -R.T
        Jx2[3:6,3:6] = -R.T
        Jx2[3:6,0:3] = R.T.dot(skew(t))
        return r, Jx1, Jx2
    else:
        return r


def pose_inv(x, calcJ = False):
    R = expSO3(x[0:3])
    t = x[3:6]
    Rinv = np.linalg.inv(R)
    xinv = np.hstack([logSO3(Rinv),Rinv.dot(-t)])
    if(calcJ == True):
        J1 = np.eye(6)
        J1[0:3,0:3] = -R
        J1[3:6,3:6] = -R
        J1[3:6,0:3] = R.dot(skew(R.T.dot(-t)))
        return xinv, J1
    else:
        return xinv

def tom(x):    
    R = expSO3(x[0:3])
    t = x[3:6]
    return makeT(R,t)

def tox(m):    
    R,t = makeRt(m)
    return np.hstack([logSO3(R),t])

if __name__ == '__main__':
    print('test pose_inv')
    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    r, J = pose_inv(x,True)
    Jm = numericalDerivative(pose_inv, [x], 0, pose_plus, pose_minus)
    ckeck(J,Jm)
 

    fx = 400.
    fy = 400.
    cx = 200.
    cy = 100.
    K = np.array([[0,fy,cy],[fx,0,cx],[0,0,1.]])
    print('test pose_plus and pose_minus')
    x1 = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    x2 = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    x3m = tox(tom(x1).dot(tom(x2)))
    x3,J1,J2 = pose_plus(x1,x2,True)
    x2m = pose_minus(x3,x1)
    ckeck(x3m,x3)
    ckeck(x2m,x2)
    print('test pose_plus error')
    J1m = numericalDerivative(pose_plus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_plus, [x1, x2], 1, pose_plus, pose_minus)
    ckeck(J1m,J1)
    ckeck(J2m,J2)
    print('test pose_minus error')
    r, J1, J2 = pose_minus(x1, x2,True)
    J1m = numericalDerivative(pose_minus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_minus, [x1, x2], 1, pose_plus, pose_minus)
    ckeck(J1m,J1)
    ckeck(J2m,J2)

    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    p = np.array([5.,6.,10.])
    pim = np.array([50.,60.])
    r,J1,J2 = transform(x, p, True)
    J1m = numericalDerivative(transform, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transform, [x, p], 1)
    print('test transform error')
    ckeck(J1m,J1)
    ckeck(J2m,J2)

    r,J1,J2 = transformInv(x, p, True)
    J1m = numericalDerivative(transformInv, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transformInv, [x, p], 1)
    print('test transformInv error')
    ckeck(J1m,J1)
    ckeck(J2m,J2)


    r, J = projection(p,K, True)
    Jm = numericalDerivative(projection,[p, K], 0)
    print('test projection error')
    ckeck(Jm,J)


    xbc = np.array([-0.1,0.3,-0.5,0.1,-0.2,0.3])
    r,J1,J2 = reporj(x, p, pim, K, xbc, True)
    J1m = numericalDerivative(reporj, [x, p, pim, K], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reporj, [x, p, pim, K], 1)
    print('test reprojection error')
    ckeck(J1m,J1)
    ckeck(J2m,J2)
    xi = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    xj = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    xbc = np.array([0.1,-0.1,-0.3,0.1,-0.1,-0.2])

    print('test getTcicj error')
    r, Jxi, Jxj = getTcicj(xi,xj,xbc,True)
    Jxim = numericalDerivative(getTcicj, [xi,xj,xbc], 0, pose_plus, pose_minus)
    Jxjm = numericalDerivative(getTcicj, [xi,xj,xbc], 1, pose_plus, pose_minus)
    ckeck(Jxi,Jxim)
    ckeck(Jxj,Jxjm)

    print('test BinvAB error')
    xa = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    xb = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    r, Jxa = BinvAB(xa,xb,True)
    Jxam = numericalDerivative(BinvAB, [xa,xb], 0, pose_plus, pose_minus)
    Jxbm = numericalDerivative(BinvAB, [xa,xb], 1, pose_plus, pose_minus)
    ckeck(Jxa,Jxam)


