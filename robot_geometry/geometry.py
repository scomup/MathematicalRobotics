import numpy as np

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
    d = np.linalg.norm(x)
    x = x/d
    p2plane = A.dot(x) + np.ones([n,1]) / d
    if(np.max(np.abs(p2plane)) > 0.2):
        return False, np.vstack([x,d]).flatten()
    else:
        return True, np.vstack([x,d]).flatten()


def point2plane(p, plane):
    d = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]
    return np.linalg.norm(d), -plane[0:3]*np.sign(d) 

def point2line(p, center, direction):
    a = center + direction * 0.1
    b = center - direction * 0.1
    pa = p - a
    pb = p - b
    ab = a - b
    pm = np.cross(pa, pb)
    ab_norm = np.linalg.norm(ab)
    pm_norm = np.linalg.norm(pm)
    d = pm_norm/ab_norm
    j = np.cross(pm, ab)/(ab_norm*pm_norm)
    return d, j


if __name__ == '__main__':
    def test_line():
        import matplotlib.pyplot as plt
        #pts = np.array([[0.1,0.2,-0.1],[1,1.02,1],[2.1,2,2.1],[2.8,3.1,3],[4.2,3.9,4]])
        pts = np.array([[0.1,0.2,-0],[1,1.02,0],[2.1,2,0],[2.8,3.0,0],[4.2,3.9,0]])
        p = np.array([1,3,3])
        s, center, direction = find_line(pts)
        if(s is False):
            return
        d, j = point2line(p, center, direction)
        r = d*j
        fig = plt.figure("line",figsize=plt.figaspect(1))
        axes = fig.add_subplot(projection='3d')
        axes.scatter(xs=pts[:,0],ys=pts[:,1],zs=pts[:,2],label='points')
        axes.scatter(xs=center[0],ys=center[1],zs=center[2],label='center')
        axes.scatter(xs=p[0],ys=p[1],zs=p[2],label='p')
        axes.plot([p[0], p[0]+r[0]],[p[1], p[1]+r[1]],[p[2], p[2]+r[2]],label='p to line')
        plt.legend()
        plt.show()
    def test_plane():
        import matplotlib.pyplot as plt
        pts = np.array([[-1,0,1.01],[1,3.02,1],[-2.1,3,1],[1,0.,1.1],[0,1,1.02]])

        p = np.array([0,0,1.5])
        s, plane = find_plane(pts)
        if(s is False):
            return

        d, j = point2plane(p, plane)
        r = d*j
        fig = plt.figure("line",figsize=plt.figaspect(1))
        axes = fig.add_subplot(projection='3d')
        axes.scatter(xs=pts[:,0],ys=pts[:,1],zs=pts[:,2],label='points')
        axes.scatter(xs=p[0],ys=p[1],zs=p[2],label='p')
        axes.plot([p[0], p[0]+r[0]],[p[1], p[1]+r[1]],[p[2], p[2]+r[2]],label='p to line')
        plt.legend()
        plt.show()
    test_plane()




