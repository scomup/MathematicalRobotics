import numpy as np
import matplotlib.pyplot as plt
from guass_newton import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.robust_kernel import *


def residual(x, param):
    a, b = param
    r = a*x[0] + x[1] - b
    j = x[0]
    return np.array([r]), np.array([[a, 1.]])


if __name__ == '__main__':
    a = np.arange(0, 8, 0.5)
    b = a*2 + 4
    b = b + np.random.normal(0, 0.6, a.shape)
    b[9] = 4
    b[13] = 6
    params = []
    for i in range(a.shape[0]):
        params.append([a[i], b[i]])

    x = np.array([0., 0])
    plt.scatter(a, b, c='black')
    gn = guassNewton(2, residual, params, None, None)
    x1 = gn.solve(x)
    gn = guassNewton(2, residual, params, None, CauchyKernel(1))
    x2 = gn.solve(x)
    gn = guassNewton(2, residual, params, None, HuberKernel(1))
    x3 = gn.solve(x)
    gn = guassNewton(2, residual, params, None, GaussianKernel(0.01))
    x4 = gn.solve(x)
    plt.plot(a, a*x1[0] + x1[1], label='None')
    plt.plot(a, a*x2[0] + x2[1], label='Cauchy')
    plt.plot(a, a*x3[0] + x3[1], label='Huber')
    plt.plot(a, a*x4[0] + x4[1], label='gaussian')
    plt.legend()
    plt.show()
