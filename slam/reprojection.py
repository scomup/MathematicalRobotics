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
        return x_cicj, Jh.dot(Jdxi), Jh.dot(Jdxj)
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

def reproj(x_wb, pw, u, K, x_bc = np.zeros(6), calcJ = False):
    """
    reporject a wolrd point to camera frame.
    """
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

def reproj2(x_wbi, x_wbj, depth, p_cj, u_i, K, x_bc = np.zeros(6), calcJ = False):
    """
    reporject a local point in camera j to camera i.
    """
    depth = float(depth)
    if(calcJ == True):
        x_cicj, dxcicj_dxwbi, dxcicj_dxwbj = getTcicj(x_wbi, x_wbj, x_bc, True)
        p_ci, dpci_dxcicj, dpci_dpcjdepth = transform(x_cicj, p_cj * depth, True)
        u_reproi, du_dpci = projection(p_ci, K, True)
        dpcjdepth_ddepth = p_cj.reshape([-1,1])
        du_ddepth = du_dpci.dot(dpci_dpcjdepth.dot(dpcjdepth_ddepth))
        du_dxbi = du_dpci.dot(dpci_dxcicj.dot(dxcicj_dxwbi))
        du_dxbj = du_dpci.dot(dpci_dxcicj.dot(dxcicj_dxwbj))
        return u_reproi-u_i, du_dxbi, du_dxbj, du_ddepth
    else:
        x_cicj = getTcicj(x_wbi, x_wbj, x_bc)
        p_ci = transform(x_cicj, p_cj * depth)
        u_reproi = projection(p_ci, K)
        return u_reproi-u_i

def reproj2_stereo(x_wbi, x_wbj, depth, p_cj, u_il, u_ir, baseline, K, x_bc = np.zeros(6), calcJ = False):
    """
    reporject a local point in camera j to camera i.
    """
    if(calcJ == True):
        x_cicj, dxcicj_dxwbi, dxcicj_dxwbj = getTcicj(x_wbi, x_wbj, x_bc, True)
        p_ci, dpci_dxcicj, dpci_dpcjdepth = transform(x_cicj, p_cj * float(depth), True)
        u_i_reproj, dui_dpci = projection(p_ci, K, True)
        u_ir_reproj, duir_dpcir = projection(p_ci - np.array([baseline,0,0]), K, True)

        dpcjdepth_ddepth = p_cj.reshape([-1,1])
        dpci_depth =  dpci_dpcjdepth.dot(dpcjdepth_ddepth)
        dpci_dxbi = dpci_dxcicj.dot(dxcicj_dxwbi)
        dpci_dxbj = dpci_dxcicj.dot(dxcicj_dxwbj)

        dui_ddepth = dui_dpci.dot(dpci_depth)
        dui_dxbi = dui_dpci.dot(dpci_dxbi)
        dui_dxbj = dui_dpci.dot(dpci_dxbj)

        duir_ddepth = duir_dpcir.dot(dpci_depth)
        duir_dxbi = duir_dpcir.dot(dpci_dxbi)
        duir_dxbj = duir_dpcir.dot(dpci_dxbj)

        r = np.hstack([u_i_reproj-u_il, u_ir_reproj-u_ir])
        J1 = np.vstack([dui_dxbi, duir_dxbi])
        J2 = np.vstack([dui_dxbj, duir_dxbj])
        J3 = np.vstack([dui_ddepth, duir_ddepth])
        return r, J1, J2, J3
    else:
        x_cicj = getTcicj(x_wbi, x_wbj, x_bc)
        p_ci = transform(x_cicj,  p_cj * depth)
        u_i_reproj = projection(p_ci, K)
        u_ir_reproj = projection(p_ci - np.array([baseline,0,0]), K)
        r = np.hstack([u_i_reproj-u_il, u_ir_reproj-u_ir])
        return r


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
    fx = 400.
    fy = 400.
    cx = 200.
    cy = 100.
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    
    print('test pose_inv')
    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    r, J = pose_inv(x,True)
    Jm = numericalDerivative(pose_inv, [x], 0, pose_plus, pose_minus)
    check(J,Jm)
    print('test pose_plus and pose_minus')
    x1 = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    x2 = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    x3m = tox(tom(x1).dot(tom(x2)))
    x3,J1,J2 = pose_plus(x1,x2,True)
    x2m = pose_minus(x3,x1)
    check(x3m,x3)
    check(x2m,x2)
    print('test pose_plus error')
    J1m = numericalDerivative(pose_plus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_plus, [x1, x2], 1, pose_plus, pose_minus)
    check(J1m,J1)
    check(J2m,J2)
    print('test pose_minus error')
    r, J1, J2 = pose_minus(x1, x2,True)
    J1m = numericalDerivative(pose_minus, [x1, x2], 0, pose_plus, pose_minus)
    J2m = numericalDerivative(pose_minus, [x1, x2], 1, pose_plus, pose_minus)
    check(J1m,J1)
    check(J2m,J2)

    print('test transform error')
    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    p = np.array([5.,6.,10.])
    r,J1,J2 = transform(x, p, True)
    J1m = numericalDerivative(transform, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transform, [x, p], 1)
    check(J1m,J1)
    check(J2m,J2)

    print('test transformInv error')
    r,J1,J2 = transformInv(x, p, True)
    J1m = numericalDerivative(transformInv, [x, p], 0, pose_plus)
    J2m = numericalDerivative(transformInv, [x, p], 1)
    check(J1m,J1)
    check(J2m,J2)


    print('test projection error')
    r, J = projection(p,K, True)
    Jm = numericalDerivative(projection,[p, K], 0)
    check(Jm,J)

    print('test getTcicj error')
    xi = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    xj = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    xbc = np.array([0.1,-0.1,-0.3,0.1,-0.1,-0.2])
    r, Jxi, Jxj = getTcicj(xi,xj,xbc,True)
    Jxim = numericalDerivative(getTcicj, [xi,xj,xbc], 0, pose_plus, pose_minus)
    Jxjm = numericalDerivative(getTcicj, [xi,xj,xbc], 1, pose_plus, pose_minus)
    check(Jxi,Jxim)
    check(Jxj,Jxjm)

    print('test BinvAB error')
    xa = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    xb = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    r, Jxa = BinvAB(xa,xb,True)
    Jxam = numericalDerivative(BinvAB, [xa,xb], 0, pose_plus, pose_minus)
    Jxbm = numericalDerivative(BinvAB, [xa,xb], 1, pose_plus, pose_minus)
    check(Jxa,Jxam)

    print('test reproj error')
    pim = np.array([50.,60.])
    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    xbc = np.array([-0.1,0.3,-0.5,0.1,-0.2,0.3])
    p = np.array([5.,6.,10.])
    r,J1,J2 = reproj(x, p, pim, K, xbc, True)
    J1m = numericalDerivative(reproj, [x, p, pim, K, xbc], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj, [x, p, pim, K, xbc], 1)
    check(J1m,J1)
    check(J2m,J2)

    print('test reproj2 error ')
    x_wbi = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    x_wbj = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    x_bc = np.array([-0.1,-0.2,00.1,0.0,0.1,-0.3])
    depth = np.array([1.5])
    p_ci = np.array([1,1,1.])
    r, J1, J2, J3 = reproj2(x_wbi, x_wbj, depth, p_ci, np.zeros(2), K, x_bc, True)
    J1m = numericalDerivative(reproj2, [x_wbi, x_wbj, depth, p_ci, np.zeros(2), K, x_bc], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj2, [x_wbi, x_wbj, depth, p_ci, np.zeros(2), K, x_bc], 1, pose_plus, delta=1e-8)
    J3m = numericalDerivative(reproj2, [x_wbi, x_wbj, depth, p_ci, np.zeros(2), K, x_bc], 2, delta=1e-8)
    check(J1,J1m)
    check(J2,J2m)
    check(J3,J3m)

    print('test reproj2_stereo error ')
    x_wbi = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    x_wbj = np.array([0.2,0.1,-0.2,-0.1,-0.2,-0.1])
    x_bc = np.array([-0.1,-0.2,00.1,0.0,0.1,-0.3])
    baseline = 0.075
    depth = np.array([1.5])
    p_cj = np.array([1,1,1.])
    reproj2_stereo(x_wbi, x_wbj, depth, p_cj, np.zeros(2),np.zeros(2),baseline, K, x_bc)
    r, J1, J2, J3 = reproj2_stereo(x_wbi, x_wbj, depth, p_cj, np.zeros(2),np.zeros(2),baseline, K, x_bc, True)
    J1m = numericalDerivative(reproj2_stereo, [x_wbi, x_wbj, depth, p_cj, np.zeros(2),np.zeros(2),baseline, K, x_bc], 0, pose_plus, delta=1e-8)
    J2m = numericalDerivative(reproj2_stereo, [x_wbi, x_wbj, depth, p_cj, np.zeros(2),np.zeros(2),baseline, K, x_bc], 1, pose_plus, delta=1e-8)
    J3m = numericalDerivative(reproj2_stereo, [x_wbi, x_wbj, depth, p_cj, np.zeros(2),np.zeros(2),baseline, K, x_bc], 2, delta=1e-8)
    check(J1,J1m)
    check(J2,J2m)
    check(J3,J3m)



