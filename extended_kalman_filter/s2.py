
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *

tolerance = 1e-5


class S2:
    def __init__(self, vec):
        self.vec = vec
        self.len = np.linalg.norm(vec)

    def boxplus(self, delta):
        Bx = self.S2_Bx()
        Bu = Bx.dot(delta)
        R = expSO3(Bu)
        return R.dot(self.vec)


    def S2_Bx(self):
        if(self.vec[0] + self.len > tolerance):
            m = self.len+self.vec[0]
            res = np.array([[ -self.vec[1], -self.vec[2] ],
                            [self.len - self.vec[1]*self.vec[1]/m, -self.vec[2]*self.vec[1]/(m)],
                            [ -self.vec[2]*self.vec[1]/m, self.len-self.vec[2]*self.vec[2]/m]])
            res /= self.len
        else:
            res = np.zeros(3,2)
            res[1, 1] = -1
            res[2, 0] = 1
        return res

a = S2(np.array([0,0,9.8]))
b = S2(np.array([0,0,9.8]))

a.boxplus(np.array([0.1,0.1]))