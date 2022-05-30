import numpy as np
import matplotlib.pyplot as plt
from guass_newton import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.robust_kernel import *

def func(a, x):
    r = a*x[0]
    j = x[0]
    return np.array([r]), np.array([[a]])

if __name__ == '__main__':
    a = np.arange(0,8, 0.5)
    b = a*2
    #b = b + np.random.normal(0,0.6,a.shape)
    b[9] = 4
    b[13] = 6    
    x = np.array([0.])
    plt.scatter(a,b, c='black')
    gn = guassNewton(a,b,func, None, CauchyKernel(1))
    x2 = gn.solve(x)
    plt.plot(a,a*x2[0],label='Cauchy')
    #plt.legend()
    plt.show()