import numpy as np

# solve y = f(a,x) - b
class guassNewton:
    def __init__(self, a, b, func, plus=None):
        self.a = a
        self.b = b
        self.func = func
        self.plus = plus

    
    def solve_once(self, x):
        H = np.zeros([x.shape[0],x.shape[0]])
        g = np.zeros([x.shape[0]])
        score = 0
        for i, a_i in enumerate(self.a):
            b_i = self.b[i]
            f_i, j_i = self.func(a_i, x)
            r_i = f_i - b_i
            H += j_i.T.dot(j_i)
            g += j_i.T.dot(r_i) 
            score += r_i.T.dot(r_i)
        H_inv = np.linalg.inv(H)
        dx = np.dot(H_inv, -g)
        return dx, score

    def solve(self, x):
        last_score = None
        x_cur = x
        while(True):   
            dx, score = self.solve_once(x_cur)
            if(self.plus is None):
                x_cur = x_cur + dx
            else:
                x_cur = self.plus(x_cur, dx)
            print(score)
            if(last_score is None):
                last_score = score
                continue
        
            if(last_score < score):
                break
            if(last_score - score < 0.0001):
                break
            last_score = score
        return x_cur


if __name__ == '__main__':
    from math_tools import *
    def func(a, x):
        r = transform3d(x, a)
        j = np.array([[1,0,0,0, a[2], -a[1]], 
                     [0,1,0,-a[2], 0, a[0]], 
                     [0,0,1,a[1], -a[0], 0]])
        return r.flatten(), j

    x = np.array([0.1,-0.1,0.1, 2.1, 2.2,-1.3])

    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    a = a.transpose()
    b = transform3d(x, a)
    b += np.random.normal(0, 0.03, (3, elements))

    gn = guassNewton(a.T,b.T,func)
    x_init = np.array([0.,0.,0., 0.,0.,0.])
    x_new = gn.solve(x_init)
