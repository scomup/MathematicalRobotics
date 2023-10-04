
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *

tolerance = 1e-5


class S2:
    def __init__(self, vec):
        self.vec = vec
        self.len = np.linalg.norm(vec)
        self.shape = [2]

    def boxplus(self, delta):
        Bx = self.S2_Bx()
        Bu = Bx.dot(delta)
        R = expSO3(Bu)
        return S2(R.dot(self.vec))

    def boxminus(self, other):
        v_sin = np.linalg.norm(skew(self.vec) @ other.vec)
        v_cos = self.vec.T @ other.vec
        theta = np.arctan2(v_sin, v_cos)
        if (v_sin < tolerance):
            if (np.abs(theta) > tolerance):
                return np.array([3.1415926, 0])
            else:
                return np.array([0, 0])
        else:
            Bx = other.S2_Bx()
            return theta/v_sin * Bx.transpose() @ skew(other.vec) @ self.vec

    def S2_Bx(self):
        if (self.vec[0] + self.len > tolerance):
            m = self.len+self.vec[0]
            res = np.array([[-self.vec[1], -self.vec[2]],
                            [self.len - self.vec[1]*self.vec[1]/m, -self.vec[2]*self.vec[1]/(m)],
                            [-self.vec[2]*self.vec[1]/m, self.len-self.vec[2]*self.vec[2]/m]])
            res /= self.len
        else:
            res = np.zeros(3, 2)
            res[1, 1] = -1
            res[2, 0] = 1
        return res
if __name__ == '__main__':
    def func(v, grav):
        return v + grav.vec

    def plus(g, delta):
        return g.boxplus(delta)

    g = S2(expSO3(np.array([0.1, 0.0, 0.0])) @ np.array([0, 0, -9.8]))

    v = np.array([0, 0, 0.8])

    Jg = numericalDerivative(func, [v, g], 1, plus)
    Jg_prime = -skew(g.vec) @ g.S2_Bx()

    delta = S2(np.array([0.1, 0.1]))

    g_prime = g.boxplus(np.array([0.1, 0.1]))
    delta_prime = g_prime.boxminus(g)
    print(delta_prime)
