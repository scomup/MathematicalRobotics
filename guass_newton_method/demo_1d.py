import numpy as np
import matplotlib.pyplot as plt
from guass_newton import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.robust_kernel import *


a = np.arange(-10, 10, 0.1)

# kernel = GaussianKernel(0.01)
kernel = PseudoHuberKernel(2)
# kernel = CauchyKernel(2)
# kernel = HuberKernel(2)
f = a*a/2
rho = kernel.apply(f)
dr = a*rho[1]
d2r = rho[1] + rho[2]*a**2


if __name__ == '__main__':
    x = -5
    dx = 1000
    while(np.abs(dx) > 0.0001):
        ro = kernel.apply(x*x/2)
        g = ro[1]*x
        H = ro[1]
        plt.cla()
        plt.plot(a, rho[0], label='rho0(f)')
        plt.plot(a, dr, label='d_rho/d_x')
        plt.plot(a, d2r, label='d2_rho/dxdx')
        plt.plot(a, -dr/d2r, label='step')
        plt.plot(a, -dr/(rho[1]), label='step2')
        plt.scatter(x, ro[0], c='red', s=100)
        plt.legend()
        plt.ylim(-1, 10)
        plt.pause(1)
        if (H == 0):
            break
        dx = -g/H
        x += dx
        print(x)
    plt.show()
