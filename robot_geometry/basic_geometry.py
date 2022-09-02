import numpy as np
from geometry_plot import *

#https://zhuanlan.zhihu.com/p/548579394


def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def find_line(pts):
    n = pts.shape[0]
    center = np.mean(pts,axis=0)
    pts_norm = pts -  np.tile(center, (n,1))
    A = pts_norm.T.dot(pts_norm)/n
    v, D = eigen(A)
    direction = D[:,0] / np.linalg.norm(D[:,0])
    if (v[0] > 3 * v[1]):
        return True, center, direction
    else:
        return False, None, None

def find_plane(pts):
    n = pts.shape[0]
    A = pts
    b = -np.ones([n,1])
    x = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(b))
    plane = np.vstack([x,1]).flatten()
    plane /= np.linalg.norm(plane[0:3])
    p2plane = A.dot(plane[0:3]) + np.ones([n,1]) *plane[3]
    if(np.max(np.abs(p2plane)) > 0.2):
        return False, plane
    else:
        return True, plane


def point2plane(p, plane):
    d = p.dot(plane[0:3]) + plane[3] #-plane[0:3]*np.sign(d) 
    r = np.linalg.norm(d)
    return  r,plane[0:3]*np.sign(d)*r

def point2line(p, center, direction, calc_J=False):
    a = center + direction * 0.1
    b = center - direction * 0.1
    pa = a - p
    pb = b - p
    ab = b - a
    pm = np.cross(pa, pb)
    ab_norm = np.linalg.norm(ab)
    pm_norm = np.linalg.norm(pm)
    d = pm_norm/ab_norm
    j = np.cross(-pm, ab)/(ab_norm*ab_norm)
    a1,a2,a3 = a
    b1,b2,b3 = b
    p1,p2,p3 = p
    j0 = ((2*a2 - 2*b2)*((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))/2 + (-2*a3 + 2*b3)*(-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))/2)/(np.sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*np.sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
    j1 = ((-2*a1 + 2*b1)*((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))/2 + (2*a3 - 2*b3)*((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))/2)/(np.sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*np.sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
    j2 = ((2*a1 - 2*b1)*(-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))/2 + (-2*a2 + 2*b2)*((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))/2)/(np.sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*np.sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
    #j = np.array([j0,j1,j2])
    if(calc_J):
        return  np.array([d]), -j
    else:
        return  np.array([d])

"""
import sympy
from sympy import diff, Matrix, Array,symbols

a1,a2,a3 = symbols('a1,a2,a3')
b1,b2,b3 = symbols('b1,b2,b3')
p1,p2,p3 = symbols('p1,p2,p3')
a = Matrix([a1,a2,a3])
b = Matrix([b1,b2,b3])
p = Matrix([p1,p2,p3])
pa = a - p
pb = b - p
ab = b - a
pm = pa.cross(pb)
ab_norm = sympy.sqrt(ab[0]**2 + ab[1]**2 + ab[2]**2)
pm_norm = sympy.sqrt(pm[0]**2 + pm[1]**2 + pm[2]**2)
d = pm_norm/ab_norm
j = (-pm.cross(ab))/(ab_norm*ab_norm)


((2*a2 - 2*b2)*((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))/2 + (-2*a3 + 2*b3)*(-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))/2)/(sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
((-2*a1 + 2*b1)*((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))/2 + (2*a3 - 2*b3)*((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))/2)/(sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
((2*a1 - 2*b1)*(-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))/2 + (-2*a2 + 2*b2)*((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))/2)/(sqrt((-a1 + b1)**2 + (-a2 + b2)**2 + (-a3 + b3)**2)*sqrt(((a1 - p1)*(b2 - p2) - (a2 - p2)*(b1 - p1))**2 + (-(a1 - p1)*(b3 - p3) + (a3 - p3)*(b1 - p1))**2 + ((a2 - p2)*(b3 - p3) - (a3 - p3)*(b2 - p2))**2))
""" 


if __name__ == '__main__':
    def test_line():
        import matplotlib.pyplot as plt
        #pts = np.array([[0.1,0.2,-0.1],[1,1.02,1],[2.1,2,2.1],[2.8,3.1,3],[4.2,3.9,4]])
        pts = np.array([[0.1,0.2,-0],[1,1.02,0],[2.1,2.5,0],[2.8,3.0,0],[4.2,3.9,0]])
        p = np.array([1,3,3])
        s, center, direction = find_line(pts)
        if(s is False):
            return
        _, r = point2line(p, center, direction, True)
        #r = d*j
        fig = plt.figure("line",figsize=plt.figaspect(1))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],label='points')
        draw_point(ax, p, 'p')
        draw_arrow(ax, p ,-r, 'p to line')
        draw_line(ax, center, direction, 'line')
        set_axes_equal(ax)
    def test_plane():
        import matplotlib.pyplot as plt
        pts = np.array([[-1,0,2.01],[1,3.02,1],[-2.1,3,1],[1,0.,1.1],[0,1,1.02]])

        p = np.array([0,0,3])
        s, plane = find_plane(pts)

        _, r = point2plane(p, plane)
        #r = d*j
        fig = plt.figure("plane",figsize=plt.figaspect(1))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs=pts[:,0],ys=pts[:,1],zs=pts[:,2],label='points')
        center = np.mean(pts,axis=0)
        draw_point(ax, p, 'p')
        draw_plane(ax, plane, center,size=[2,2])
        draw_arrow(ax, p ,-r, 'p to plane')
        set_axes_equal(ax)
    test_line()
    test_plane()
    plt.show()





