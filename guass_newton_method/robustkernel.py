import numpy as np
import matplotlib.pyplot as plt
# Christopher Zach. “Robust bundle adjustment revisited” (11)(12)

def L2Kernel(e2):
    rho = [None,None,None]
    rho[0] = e2
    rho[1] = 1.
    rho[2] = 0
    return rho

def L1Kernel(e2):
    rho = [None,None,None]
    rho[0] = np.sqrt(e2)
    rho[1] = 1/(2*np.sqrt(e2))
    rho[2] = 0
    return rho

def HuberKernel(e, _delta = 2):
    rho = [None,None,None]
    dsqr = _delta * _delta
    if (e <= dsqr):
        rho[0] = e
        rho[1] = 1.
        rho[2] = 0.
    else:# outlier
        sqrte = np.sqrt(e);# absolut value of the error
        rho[0] = 2 * sqrte * _delta - dsqr# rho(e)   = 2 * delta * e^(1/2) - delta^2
        rho[1] = _delta / sqrte# rho'(e)  = delta / sqrt(e)
        rho[2] = -0.5 * rho[1]/e
        # rho''(e) = -1 / (2*e^(3/2)) = -1/2 * (delta/e) / e
    return rho



def PseudoHuberKernel(e2, _delta = 2):
    rho = [None,None,None]
    dsqr = _delta * _delta
    dsqrReci = 1. / dsqr
    aux1 = dsqrReci * e2 + 1.0
    aux2 = np.sqrt(aux1)
    rho[0] = (2 * dsqr * (aux2 - 1))
    rho[1] = 1. / aux2
    rho[2] = -0.5 * dsqrReci * rho[1] / aux1;   
    return rho


def CauchyKernel(e2, _delta = 2):
    rho = [None,None,None]
    c2 = _delta * _delta
    c2_inv = 1. / c2
    aux = c2_inv * e2 + 1.0
    rho[0] =(c2 * np.log(aux))
    rho[1] =(1. / aux)
    rho[2] = -(c2_inv * rho[1] * rho[1])
    return rho

def solveR(a, b, kernel):
    x = np.array([[0.],[0.]])
    J = np.zeros([2,a.shape[0]])
    J[0,:] = a
    J[1,:] = 1.
    J = J.T
    cont = 0
    while(True):
        cost = 0.
        error = a*x[0] + x[1] - b
        H = np.zeros([2,2])
        g = np.zeros([2,1])
        for i in range(a.shape[0]):
            error2 = error[i]*error[i]
            rho = kernel(error2)
            j = J[i].reshape(1,2)
            w = rho[1]  #+ 2*rho[2] * error2
            #n = np.linalg.norm(w)
            #print(n)
            H += j.T.dot(j * w)
            g += j.T.dot(error[i] * rho[1])
            cost += rho[0]
        cont += 1
        dx = np.linalg.solve(H, -g)
        #print(dx)
        x = x + dx
        if(np.linalg.norm(dx)<0.00001):
            break
    #print(cont)
    return x


def drawKernel():
    e = np.arange(-10,10, 0.1)
    e2 = e**2
    eHuber = PseudoHuberKernel(e2)
    eCauchy = CauchyKernel(e2)
    eL1 = L1Kernel(e2)
    plt.plot(e, eHuber[0], label='HuberKernel')
    plt.plot(e, eCauchy[0], label='CauchyKernel')
    plt.plot(e, e2, label='L2')
    plt.plot(e, eL1[0], label='L1')
    plt.legend()
    plt.show()

#drawKernel()

x0 = 2.
x1 = 4.
a = np.arange(0,8, 0.5)
b = a*x0 + x1
b = b + np.random.normal(0,0.6,a.shape)
b[9] = 4
b[13] = 6
x0 = solveR(a, b, L2Kernel)
x1 = solveR(a, b, PseudoHuberKernel)
x2 = solveR(a, b, CauchyKernel)
x3 = solveR(a, b, L1Kernel)


plt.scatter(a,b)
plt.plot(a,a*x0[0] + x0[1],label='L2')
plt.plot(a,a*x1[0] + x1[1],label='HuberKernel')
plt.plot(a,a*x2[0] + x2[1],label='CauchyKernel')
plt.plot(a,a*x3[0] + x3[1],label='L1')
plt.legend()


plt.show()