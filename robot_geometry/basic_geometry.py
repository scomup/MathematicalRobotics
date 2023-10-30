import numpy as np
from geometry_plot import *

# https://zhuanlan.zhihu.com/p/548579394


def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)


def find_line(pts):
    n = pts.shape[0]
    center = np.mean(pts, axis=0)
    pts_norm = pts - np.tile(center, (n, 1))
    A = pts_norm.T.dot(pts_norm)/n
    v, D = eigen(A)
    direction = D[:, 0] / np.linalg.norm(D[:, 0])
    if (v[0] > 3 * v[1]):
        return True, center, direction
    else:
        return False, None, None


def find_plane(pts):
    n = pts.shape[0]
    A = pts
    b = -np.ones([n, 1])
    x = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(b))
    plane = np.vstack([x, 1]).flatten()
    plane /= np.linalg.norm(plane[0:3])
    p2plane = A.dot(plane[0:3]) + np.ones([n, 1]) * plane[3]
    if (np.max(np.abs(p2plane)) > 0.2):
        return False, plane
    else:
        return True, plane


def point2plane(p, plane, calc_J=False):
    d = p.dot(plane[0:3]) + plane[3]  # -plane[0:3]*np.sign(d)
    if (calc_J):
        return d, plane[0:3]
    else:
        return d


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
    if (calc_J):
        j = np.cross(pm, ab)/(pm_norm*ab_norm)
        return np.array([d]), j
    else:
        return np.array([d])


if __name__ == '__main__':
    def test_line():
        import matplotlib.pyplot as plt
        # pts = np.array([[0.1, 0.2, -0.1], [1, 1.02, 1], [2.1, 2, 2.1], [2.8, 3.1, 3], [4.2, 3.9, 4]])
        pts = np.array([[0.1, 0.2, -0], [1, 1.02, 0], [2.1, 2.5, 0], [2.8, 3.0, 0], [4.2, 3.9, 0]])
        p = np.array([1, 3, 3])
        s, center, direction = find_line(pts)
        if (s is False):
            return
        r, j = point2line(p, center, direction, True)
        g = -j*r
        fig = plt.figure("line", figsize=plt.figaspect(1))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], label='points')
        draw_point(ax, p, 'p')
        draw_arrow(ax, p, g, 'p to line')
        draw_line(ax, center, direction, 'line')
        set_axes_equal(ax)

    def test_plane():
        import matplotlib.pyplot as plt
        pts = np.array([[-1, 0, 2.01], [1, 3.02, 1], [-2.1, 3, 1], [1, 0., 1.1], [0, 1, 1.02]])

        p = np.array([0, 0, 3])
        s, plane = find_plane(pts)

        r, j = point2plane(p, plane, True)
        g = -j*r
        fig = plt.figure("plane", figsize=plt.figaspect(1))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs=pts[:, 0], ys=pts[:, 1], zs=pts[:, 2], label='points')
        center = np.mean(pts, axis=0)
        draw_point(ax, p, 'p')
        draw_plane(ax, plane, center, size=[2, 2])
        draw_arrow(ax, p, g, 'p to plane')
        set_axes_equal(ax)
    test_line()
    test_plane()
    plt.show()
