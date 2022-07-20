import numpy as np
import sympy
from sympy import diff, Matrix, Array,symbols
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *

"""
fx,fy,cx,cy = symbols('fx,fy,cx,cy')
x,y,z = symbols('x,y,z')

def H(p):
    u = (p[0] * fx + cx * p[2])/p[2]
    v = (p[1] * fy + cy * p[2])/p[2]
    return Matrix([[u],[v]])

p = Matrix([x,y,z])
J = Matrix([[diff(H(p),x)],[diff(H(p),y)],[diff(H(p),z)]]).reshape(3,2).T
"""
fx = 400.
fy = 400.
cx = 200.
cy = 100.
def reporj(x,pw, calcJ = False):
    R = expSO3(x[0:3])
    t = x[3:6]
    pc = R.dot(pw) + t
    u = (pc[0] * fx + cx * pc[2])/pc[2]
    v = (pc[1] * fy + cy * pc[2])/pc[2]
    r = np.array([u,v])
    if(calcJ == True):
        dHdT = np.array([[fx/pc[2],    0, cx/pc[2] - (cx*pc[2] + fx*pc[0])/pc[2]**2],
                         [   0, fy/pc[2], cy/pc[2] - (cy*pc[2] + fy*pc[1])/pc[2]**2]])
        M = R.dot(skew(-pw))
        dTdx = np.hstack([M, R])
        dTdp = R
        return  r, dHdT.dot(dTdx), dHdT.dot(dTdp)
    else:
        return r


def plus(x1,x2):
    R1 = expSO3(x1[0:3])
    t1 = x1[3:6]
    R2 = expSO3(x2[0:3])
    t2 = x2[3:6]
    R = R1.dot(R2)
    t = R1.dot(t2) + t1
    return np.hstack([logSO3(R),t])

if __name__ == '__main__':
    x = np.array([0.1,0.3,0.5,0.1,0.2,0.3])
    plus(x,x)
    p = np.array([0.1,0.2,0.3])
    r,J1,J2 = reporj(x,p, True)
    J1m = numericalDerivative(reporj,[x,p],0,plus)
    J2m = numericalDerivative(reporj,[x,p],1)
    print('test reprojection error')
    if(np.linalg.norm(J1m - J1) < 0.01):
        print('OK')
    else:
        print('NG')
    if(np.linalg.norm(J2m - J2) < 0.01):
        print('OK')
    else:
        print('NG')
    