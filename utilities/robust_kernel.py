import numpy as np
# Christopher Zach. “Robust bundle adjustment revisited” (11)(12)


class GaussianKernel:
    def __init__(self, d):
        self.d = d

    def apply(self, e2):
        rho = [None, None, None]
        t = np.exp(-self.d*e2)
        rho[0] = 1-t
        rho[1] = t * self.d
        rho[2] = - self.d * rho[1]
        return rho


class L2Kernel:
    def __init__(self):
        pass

    def apply(self, e2):
        rho = [None, None, None]
        rho[0] = e2
        rho[1] = 1.
        rho[2] = 0
        return rho


class L1Kernel:
    def __init__(self):
        pass

    def apply(self, e2):
        rho = [None, None, None]
        rho[0] = np.sqrt(e2)
        rho[1] = 1/(2*np.sqrt(e2))
        rho[2] = 0
        return rho


class L4Kernel:
    def __init__(self):
        pass

    def apply(self, e2):
        rho = [None, None, None]
        rho[0] = e2*e2
        rho[1] = 2*e2
        rho[2] = 2
        return rho


class HuberKernel:
    def __init__(self, delta=2):
        self.delta = delta

    def apply(self, e2):
        rho = [None, None, None]
        dsqr = self.delta * self.delta
        if (e2 <= dsqr):
            rho[0] = e2
            rho[1] = 1.
            rho[2] = 0.
        else:
            sqrte = np.sqrt(e2)
            rho[0] = 2 * sqrte * self.delta - dsqr
            rho[1] = self.delta / sqrte
            rho[2] = -0.5 * rho[1] / e2
        return rho


class PseudoHuberKernel:
    def __init__(self, delta=2):
        self.delta = delta

    def apply(self, e2):
        rho = [None, None, None]
        dsqr = self.delta * self.delta
        dsqrReci = 1. / dsqr
        aux1 = dsqrReci * e2 + 1.0
        aux2 = np.sqrt(aux1)
        rho[0] = (2 * dsqr * (aux2 - 1))
        rho[1] = 1. / aux2
        rho[2] = -0.5 * dsqrReci * rho[1] / aux1
        return rho


class CauchyKernel:
    def __init__(self, delta=2):
        self.delta = delta

    def apply(self, e2):
        rho = [None, None, None]
        c2 = self.delta * self.delta
        c2_inv = 1. / c2
        aux = c2_inv * e2 + 1.0
        rho[0] = (c2 * np.log(aux))
        rho[1] = (1. / aux)
        rho[2] = -(c2_inv * rho[1] * rho[1])
        return rho


def drawKernel():
    import matplotlib.pyplot as plt
    e = np.arange(-5, 5, 0.03)
    e2 = e**2
    ePHuber = PseudoHuberKernel(1).apply(e2)
    # eHuber = HuberKernel(1).apply(e2)
    eCauchy = CauchyKernel(1).apply(e2)
    # eL1 = L1Kernel().apply(e2)
    eGaussian = GaussianKernel(1).apply(e2)
    # plt.plot(e, eHuber[0], label='Huber')
    plt.plot(e, ePHuber[0], label='PseudoHuber')
    plt.plot(e, eCauchy[0], label='Cauchy')
    plt.plot(e, e2, label='L2')
    # plt.plot(e, eL1[0], label='L1')
    plt.plot(e, eGaussian[0], label='Gaussian')
    plt.legend()
    plt.ylim(0, 5)
    plt.show()

if __name__ == '__main__':
    drawKernel()
