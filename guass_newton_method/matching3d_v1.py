import numpy as np
from math_tools import *
from guass_newton import *
from matching3d import plot3D


def func(a, x):
    R, t = makeRt(p2m(x))
    r = R.dot(a) + t
    j = np.array([[1,0,0,0, a[2], -a[1]], 
                 [0,1,0,-a[2], 0, a[0]], 
                 [0,0,1,a[1], -a[0], 0]])
    j = R.dot(j)
    return r.flatten(), j

def plus(x1, x2):
    return m2p(p2m(x1).dot(p2m(x2)))

    
if __name__ == '__main__':

    plt3d = plot3D()
    x_truth = np.array([1000,-0.1,0.1, 2.1, 2.2,-1.3])
    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    b = transform3d(x_truth, a.T,p2m).T
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
        cur_a = transform3d(x_cur, a.T,p2m).T
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