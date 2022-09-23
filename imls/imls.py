import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import numpy as np

def eigen(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def find_normal(pts):
    n = pts.shape[0]
    center = np.mean(pts,axis=0)
    pts_norm = pts -  np.tile(center, (n,1))
    A = pts_norm.T.dot(pts_norm)/n
    v, D = eigen(A)
    direction = D[:,1] / np.linalg.norm(D[:,1])
    if (v[0] > 3 * v[1]):
        return True, direction
    else:
        return False, None

def get_norm_vec(pt):
    norm_vec = np.zeros([pt.shape[0],2])
    tree = KDTree(pt)
    print(norm_vec)
    for i, p in enumerate(pt):
        print(p)
        dists, indexes = tree.query(p, k=5)
        near_pt = pt[indexes]
        s, vec = find_normal(near_pt)
        #if s is False:
        #    continue
        norm_vec[i,:] = vec
        print(indexes)
    return norm_vec

def p2surf(x, pt, nvec):
    h = 1
    tree = KDTree(pt)
    indexes = tree.query_ball_point(x, h)
    near_pt = pt[indexes]
    weight_sum = 0.
    proj_sum = 0.
    for i, p in enumerate(near_pt):
        e_height = x - p
        weight  = np.exp(-(e_height.dot(e_height)/(h*h)))
        proj_sum += weight *e_height * nvec[i]
        weight_sum += weight
    height = proj_sum / weight_sum
    return height

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

f = lambda x: np.cos(x)
xx = np.linspace(-2*np.pi, 2*np.pi, 30)
yy = f(xx)
ax.scatter(xx, yy)
pt = np.vstack([xx,yy]).T
nvec = get_norm_vec(pt)

x = np.array([1.6,0.8])
ax.scatter(x[0], x[1],c='r')

height = p2surf(x, pt, nvec)
for i, p in enumerate(pt):
    ax.annotate('', xy=p + nvec[i,:] * 0.2, xytext=p,
                arrowprops=dict(arrowstyle='-|>', 
                                connectionstyle='arc3', 
                                facecolor='C0', 
                                edgecolor='C0')
               )

ax.annotate('', xy=x - height, xytext=x,
            arrowprops=dict(arrowstyle='-|>', 
                            connectionstyle='arc3', 
                            facecolor='C0', 
                            edgecolor='C0')
           )


plt.grid()
plt.show()