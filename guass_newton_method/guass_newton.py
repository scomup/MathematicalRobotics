import numpy as np

# solve y = f(a,x) - b
class guassNewton:
    """
    A guass newton solver.
    more information is written in guass_newton_method.md
    """
    def __init__(self, a, b, residual, plus=None, kernel=None):

        self.a = a
        self.b = b
        self.residual = residual
        self.plus = plus
        self.kernel = kernel

    
    def solve_once(self, x):
        H = np.zeros([x.shape[0],x.shape[0]])
        g = np.zeros([x.shape[0]])
        score = 0
        for i, a_i in enumerate(self.a):
            b_i = self.b[i]
            r_i, j_i = self.residual(x, a_i, b_i)
            e2 = r_i.T.dot(r_i)
            if(self.kernel is None):
                H += j_i.T.dot(j_i)
                g += j_i.T.dot(r_i) 
                score += e2
            else:
                rho = self.kernel.apply(e2)
                df = j_i.T.dot(r_i)
                g += rho[1]*j_i.T.dot(r_i) 
                #df = df.reshape([df.shape[0],1])
                H += rho[1]*j_i.T.dot(j_i) #+ rho[2] * df.dot(df.T) 
                score += rho[0]
        dx = np.linalg.solve(H, -g)
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
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utilities.math_tools import *

    def residual(x, a, b):
        r = transform3d(x, a).flatten() - b
        j = np.array([[1,0,0,0, a[2], -a[1]], 
                     [0,1,0,-a[2], 0, a[0]], 
                     [0,0,1,a[1], -a[0], 0]])
        return r, j

    x = np.array([0.1,-0.1,0.1, 2.1, 2.2,-1.3])

    elements = 100
    a = (np.random.rand(elements,3)-0.5)*2
    a = a.transpose()
    b = transform3d(x, a)
    b += np.random.normal(0, 0.03, (3, elements))

    gn = guassNewton(a.T,b.T,residual)
    x_init = np.array([0.,0.,0., 0.,0.,0.])
    x_new = gn.solve(x_init)
