#!/usr/bin/env python3
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.plot_tools import *


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


def expSO2(v):
    return np.array([[np.cos(v), -np.sin(v)],
                     [np.sin(v), np.cos(v)]])


def logSO2(m):
    return np.arctan2(m[1, 0], m[0, 0])


def hat2d(v):
    return np.array([-(v)[1], (v)[0]])


def boxplus(a, b):
    p = a[0:2] + b[0:2]
    R = expSO2(a[2]).dot(expSO2(b[2]))
    v = a[3:5] + b[3:5]
    return np.array([p[0], p[1], logSO2(R), v[0], v[1]])


def boxminus(a, b):
    p = a[0:2] - b[0:2]
    R = expSO2(b[2]).T.dot(expSO2(a[2]))
    v = a[3:5] - b[3:5]
    return np.array([p[0], p[1], logSO2(R), v[0], v[1]])


def func(x, u, dt, n=np.array([0, 0, 0])):
    p = x[0:2] + x[3:5] * dt
    R = expSO2(x[2]).dot(expSO2((u[2] - n[2]) * dt))
    v = expSO2(x[2]).dot(u[0:2] - n[0:2])
    return np.array([p[0], p[1], logSO2(R), v[0], v[1]])


def F(x_last_hat, x_last_delta, u, n, dt):
    x_curr_true = func(boxplus(x_last_hat, x_last_delta), u, dt, n)
    x_curr_hat = func(x_last_hat, u, dt)
    x_curr_delta = boxminus(x_curr_true, x_curr_hat)
    return x_curr_delta

x_last_hat = np.array([1, 2, 0.5, 0, 2])
x_last_delta = np.array([0, 0, 0, 0, 0])
u = np.array([5, 2, 3])
n = np.array([0, 0, 0])
dt = 0.1
res = F(x_last_hat, x_last_delta, u, n, dt)


def get_Jx(x, u, dt):
    Jx = np.zeros([5, 5])
    Jx[0:2, 0:2] = np.eye(2)
    Jx[0:2, 3:5] = np.eye(2)*dt
    Jx[2, 2] = 1
    Jx[3:5, 2] = expSO2(x[2]).dot(hat2d(u[0:2]))
    return Jx
get_Jx(x_last_hat, u, dt)


def get_Jn(x, u, dt):
    Jn = np.zeros([5, 3])
    Jn[2, 2] = -dt
    Jn[3:5, 0:2] = -expSO2(x[2])
    return Jn


Jx = numericalDerivative(F, [x_last_hat, x_last_delta, u, n, dt], 1, plus=boxplus, minus=boxminus)
"""
I_2x2     0         I_2x2 dt
0        1         0
0_2x2 exp(th)hat2d(u_v-n_v) 0
"""
Jn = numericalDerivative(F, [x_last_hat, x_last_delta, u, n, dt], 3, minus=boxminus)
"""
0_2x2     0_2x1
0_2x1     -dt
-exp(th)  0_2x1
"""
figname = "test"
fig = plt.figure(figname)
axes = fig.gca()

P = np.zeros([5, 5])
P[0:2, 0:2] = np.eye(2) * 0.1
P[2, 2] = 0.0
Q = np.zeros([3, 3])
Q[0, 0] = 0.2
Q[1, 1] = 0.01
Q[2, 2] = 0.001

get_Jn(x_last_hat, u, dt)
x = np.array([0, 0, 0, 0, 0])
u = np.array([2, 0, 0.2])
dt = 1.
for i in range(10):
    x = func(x, u, dt)
    Jn = get_Jn(x, u, dt)
    Jx = get_Jx(x, u, dt)
    # Jx = numericalDerivative(F, [x, x_last_delta, u, n, dt], 1, plus=boxplus, minus=boxminus)
    # Jn = numericalDerivative(F, [x, x_last_delta, u, n, dt], 3, minus=boxminus)

    P = Jx.dot(P.dot(Jx.T)) + Jn.dot(Q.dot(Jn.T))
    # zprint(P[0:2, 0:2])
    plot_pose2(figname, v2m(x[0:3]), 0.2, P[0:2, 0:2])

plt.show()
