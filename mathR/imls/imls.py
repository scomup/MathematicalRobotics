import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np

def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)

def find_normal(pts):
    n = pts.shape[0]
    center = np.mean(pts, axis=0)
    pts_norm = pts -  np.tile(center, (n, 1))
    A = pts_norm.T.dot(pts_norm)/n
    v, D = eigen(A)
    direction = D[:, 1] / np.linalg.norm(D[:, 1])
    if (v[0] > 3 * v[1]):
        return True, direction
    else:
        return False, None

def get_norm_vec(pt):
    norm_vec = np.zeros([pt.shape[0], 2])
    tree = KDTree(pt)
    for i, p in enumerate(pt):
        #_, indexes = tree.query(p, k=5)
        indexes = tree.query_ball_point(p, 1)
        near_pt = pt[indexes]
        s, vec = find_normal(near_pt)
        norm_vec[i,:] = vec
    return norm_vec

def p2surf(x, pt, n):
    n = n.reshape([-1, 2])
    pt = pt.reshape([-1, 2])

    h = 1
    tree = KDTree(pt)
    # indexes = tree.query_ball_point(x, h)
    dists, indexes = tree.query(x, k=3)
    # near_pt = pt[indexes]
    weight_sum = 0.
    proj_sum = 0.
    dir_sum = np.array([0, 0.])
    for i in indexes:
        if (i > pt.shape[0]):
            continue
        p = pt[i]
        x_p = x - p
        weight  = np.exp(-(x_p.dot(x_p)/(h*h)))
        weight = 1.
        sign = np.sign(x_p.dot(n[i]))
        proj_sum += weight * x_p.dot(n[i]) * sign
        weight_sum += weight
        dir_sum += weight * n[i] * sign
        break
    dist = proj_sum / weight_sum
    dir = dir_sum / weight_sum
    return dist, dir

def draw_norm(figname, pt, n, c='C0'):
    pt = pt.reshape([-1, 2])
    n = n.reshape([-1, 2])
    fig = plt.figure(figname)
    axes = fig.gca()
    for i, p in enumerate(pt):
        axes.annotate('', xy=p + n[i], xytext=p,
                      arrowprops=dict(arrowstyle='-|>',
                      connectionstyle='arc3',
                      facecolor=c,
                      edgecolor=c))


def test_data():
    f = lambda x: np.cos(1.5*x)
    xx = np.linspace(-np.pi, np.pi, 50)
    yy = f(xx)
    pt = np.vstack([xx, yy]).T
    pt += np.random.normal(0, 0.05, pt.shape)

    xx = np.linspace(-np.pi, np.pi, 100)
    yy = f(xx)
    truth = np.vstack([xx, yy]).T
    return pt, truth

pt, truth = test_data()
nvec = get_norm_vec(pt)

# x = np.array([2.0, 0.0])
# x = np.array([0.2, 0.6])
xs = np.array([[2, -0.5], [1.0, 0.4], [0.2, 0.6]])


fig = plt.figure("test")
ax = fig.gca()
ax.set_aspect('equal')

"""
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)

n = np.array([0.7, 1.])
n /= np.linalg.norm(n)
p = np.array([0, 0])
x = np.array([1, 1])
dist, dir = p2surf(x, p, n)
v = - dist * dir
ax.scatter(x[0], x[1], c='r')
ax.scatter(p[0], p[1], c='C0')
draw_norm('test', x, v, 'r')
draw_norm('test', p, n, 'C0')
"""

ax.scatter(pt[:, 0], pt[:, 1])
# draw_norm('test', pt, nvec * 0.3)

for x in xs:
    dist, dir = p2surf(x, pt, nvec)
    v = - dist * dir
    ax.scatter(x[0], x[1], c='r')
    draw_norm('test', x, v, 'r')


ax.plot(truth[:, 0], truth[:, 1])

plt.grid()
plt.show()
