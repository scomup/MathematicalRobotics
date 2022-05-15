import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math_tools import *
from guass_newton import *

class plot3D:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
    def update(self,a,b):
        plt.cla()
        self.ax.scatter3D(a[:,0],a[:,1],a[:,2], c= 'r')
        self.ax.scatter3D(b[:,0],b[:,1],b[:,2], c= 'b')
        plt.pause(0.1)
    def show(self):
        plt.show()

def func(a, x):
    R, t = makeRt(expSE3(x))
    r = R.dot(a) + t
    M = R.dot(skew(-a))
    j = np.hstack([R,M])
    return r.flatten(), j

def plus(x1, x2):
    return logSE3(expSE3(x1).dot(expSE3(x2)))
    
if __name__ == '__main__':

    plt3d = plot3D()
    x_truth = np.array([1000,-0.1,0.1, 2.1, 2.2,-1.3])
    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    b = transform3d(x_truth, a.T).T
    b += np.random.normal(0, 0.03, (elements,3))
    gn = guassNewton(a,b,func, plus)
    x_cur = np.array([0.,0.,0., 0.,0.,0.])
    cur_a = a.copy()
    last_score = None
    iter = 0
    while(True):      
        plt3d.update(cur_a, b)
        dx, score = gn.solve_once(x_cur)
        x_cur = gn.plus(x_cur, dx)
        cur_a = transform3d(x_cur, a.T).T
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
    plt3d.show()