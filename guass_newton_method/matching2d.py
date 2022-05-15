import numpy as np
import matplotlib.pyplot as plt
from math_tools import *
from guass_newton import *
from robust_kernel import *

class plot2D:
    def __init__(self):
        #self.fig = plt.figure()
        self.fig, self.ax = plt.subplots()
    def update(self,a,b):
        plt.cla()
        self.ax.scatter(a[:,0],a[:,1], c= 'r')
        self.ax.scatter(b[:,0],b[:,1], c= 'b')
        plt.pause(0.1)
    def show(self):
        plt.show()

def func(a, x):
    R, t = makeRt(v2m(x))
    r = R.dot(a) + t
    j = np.array([[1, 0, -a[1]],
                  [0, 1,  a[0]]])
    j = R.dot(j)
    return r.flatten(), j

def plus(x1, x2):
    return m2v(v2m(x_cur).dot(v2m(dx)))

if __name__ == '__main__':

    plt2d = plot2D()
    x_truth = np.array([-0.3,0.2,np.pi/2])
    elements = 10
    #a = (np.random.rand(elements,2)-0.5)*2

    a = np.array([[0,0],[0,0.5],[0, 1], [0.5, 0], [0.5, 1], [1,0],[1,0.5],[1,1]  ])
    b = transform2d(x_truth, a.T).T
    b[0,1] -= 0.5
    b[0,0] += 0.5
    
    #b += np.random.normal(0, 0.03, (elements,2))
    kernel = CauchyKernel(0.1)
    gn = guassNewton(a,b,func, plus, kernel)
    x_cur = np.array([0.,0.,0.])
    cur_a = a.copy()
    last_score = None
    iter = 0
    while(True):      
        plt2d.update(cur_a, b)
        dx, score = gn.solve_once(x_cur)
        x_cur = gn.plus(x_cur, dx)
        cur_a = transform2d(x_cur, a.T).T
        print('iter %d: %f'%(iter, score))
        iter += 1
        if(last_score is None):
            last_score = score
            continue
        if(last_score < score):
            break
        if(last_score - score < 0.0001):
            break
        last_score = score
    plt2d.show()