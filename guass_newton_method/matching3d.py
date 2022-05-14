import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math_tools import *
from guass_newton import *

def func(a, x):
    r = transform3d(x, a)
    R = expSO3(x[3:6])
    j = np.array([[1,0,0,0, a[2], -a[1]], 
                 [0,1,0,-a[2], 0, a[0]], 
                 [0,0,1,a[1], -a[0], 0]])
    j = R.dot(j)
    return r.flatten(), j


if __name__ == '__main__':

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_truth = np.array([0.1,-0.1,0.1, 2.1, 2.2,-1.3])
    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    b = transform3d(x_truth, a.T).T
    b += np.random.normal(0, 0.03, (elements,3))

    gn = guassNewton(a,b,func)
    x_cur = np.array([0.,0.,0., 0.,0.,0.])
    cur_a = transform3d(x_cur, a.T).T
    last_score = None
    while(True):   
        
        plt.cla()
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        ax.scatter3D(cur_a[:,0],cur_a[:,1],cur_a[:,2], c= 'r')
        ax.scatter3D(b[:,0],b[:,1],b[:,2], c= 'b')
        plt.pause(0.1)
        dx, score = gn.solve_once(x_cur)
        x_cur = m2p(p2m(x_cur).dot(p2m(dx)))
        cur_a = transform3d(x_cur, a.T).T
        print(score)
        if(last_score is None):
            last_score = score
            continue
        if(last_score < score):
            break
        if(last_score - score < 0.0001):
            break
        last_score = score
    plt.show()