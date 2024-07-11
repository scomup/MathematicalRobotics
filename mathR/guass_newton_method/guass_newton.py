import numpy as np


class guassNewton:
    """
    A guass newton solver.
    more information is written in guass_newton_method.md
    """
    def __init__(self, x_size, residual, params, plus=None, kernel=None):
        self.residual = residual
        self.plus = plus
        self.kernel = kernel
        self.params = params
        self.x_size = x_size

    def solve_once(self, x):
        H = np.zeros([self.x_size, self.x_size])
        g = np.zeros([self.x_size])
        score = 0
        for i, param in enumerate(self.params):
            r_i, j_i = self.residual(x, param)
            e2 = r_i.T @ r_i
            if (self.kernel is None):
                H += j_i.T @ j_i
                g += j_i.T @ r_i
                score += e2
            else:
                rho = self.kernel.apply(e2)
                g += rho[1] * j_i.T @ r_i
                H += rho[1] * j_i.T @ j_i
                score += rho[0]
        try:
            # dx = -cho_solve(cho_factor(H), g)
            dx = -np.linalg.solve(H, g)
        except:
            print('Bad Hassian matrix!')
            dx = -np.linalg.pinv(H) @ g
        return dx, score

    def solve(self, x, step=0):
        last_score = None
        x_cur = x
        while(True):
            dx, score = self.solve_once(x_cur)
            if (step > 0 and np.max(dx) > step):
                dx = dx / np.max(dx) * step

            if (self.plus is None):
                x_cur = x_cur + dx
            else:
                x_cur = self.plus(x_cur, dx)
            print(score)
            if (last_score is None):
                last_score = score
                continue

            if (last_score < score):
                break
            if (last_score - score < 0.0001):
                break
            last_score = score
        return x_cur

