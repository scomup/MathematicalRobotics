from load_bal_datasets import BALLoader
from reprojection import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import lapack, solve
from scipy.sparse.linalg import spsolve_triangular
from os import environ
environ['OMP_NUM_THREADS'] = '1'

inds_cache = {}


class CameraVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 6)

    def update(self, dx):
        self.x = self.x @ p2m(dx)


class Pose3dEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(6), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE3 is given in a graph_optimization.md (18)(19)
        """
        Tzx = np.linalg.inv(self.z) @ vertices[self.link[0]].x
        return m2p(Tzx), [np.eye(6)]


class PointVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 3)

    def update(self, dx):
        self.x = self.x + dx


class ReprojEdge(BaseEdge):
    def __init__(self, link, z, omega=np.eye(2), kernel=None):
        super().__init__(link, z, omega, kernel)

    def residual(self, vertices):
        """
        The proof of Jocabian of SE2 is given in a graph_optimization.md (13)(14)
        """
        Tcw = vertices[self.link[0]].x
        pw = vertices[self.link[1]].x
        u, K = self.z
        r, JTcw, Jpw = reproj0(Tcw, pw, u, K, True)
        return r, [JTcw, Jpw]
    

def undistort_point(u, K, dist_coeffs):
    # Convert to homogeneous coordinates
    u_homo = np.array([u[0], u[1], 1])
    k1 = dist_coeffs[0]
    k2 = dist_coeffs[1]
    # Normalize coordinates
    p = np.linalg.inv(K) @ u_homo

    # Apply distortion correction
    r_squared = np.sum(p[:2]**2)
    """
    The camera model is described in:
    https://grail.cs.washington.edu/projects/bal/
    """
    radial_correction = 1 + k1 * r_squared + k2 * r_squared**2

    u_corrected = np.array([
        p[0] * radial_correction,
        p[1] * radial_correction
    ])

    # Convert back to homogeneous coordinates
    u_undist_homo = np.array([u_corrected[0], u_corrected[1], 1])

    # Project back to image coordinates
    u_undist = (K @ u_undist_homo)[:2]

    return u_undist

from gui import *

if __name__ == '__main__':  
    """
    cameras = np.load("cameras.npy")
    points = np.load("points.npy")
    app = QApplication([])
    R = expSO3(np.array([np.pi/2, 0, 0]))
    T = makeT(R, np.zeros(3))
    axis = GLAxisItem(size=[1, 1, 1], width=2)
    points = (R @ points.T).T
    points_item = gl.GLScatterPlotItem(pos=points, size=0.01, color=(1,1,1,0.3), pxMode=False)
    items = [points_item]
    for pose in cameras:
        cam_item = GLCameraFrameItem(size=0.2, width=2)
        axis_item = GLAxisItem(size=[0.1, 0.1, 0.1], width=2)
        new_T = T @ np.linalg.inv(p2m(pose))
        cam_item.setTransform(new_T)
        axis_item.setTransform(new_T)
        items.append(cam_item)
        items.append(axis_item)
    window = Gui3d(static_obj=items)
    window.show()
    app.exec_()
    """

    """
    H = np.load("H.npy")
    g = np.load("g.npy")
    start = time.time()

    H += np.eye(H.shape[0])
    # L = np.linalg.cholesky(H)
    # x = cho_solve((L,True), g)

    # x = np.linalg.solve(H, -g)
    # dx = -cho_solve(cho_factor(H), g)

    #lu, piv = lu_factor(H)
    #x = lu_solve((lu, piv), g)

    #x = -np.linalg.inv(H) @ g 

    #H += np.eye(H.shape[0])
    #x = fast_positive_definite_inverse(H) @ g
    #H += np.eye(H.shape[0])
    #x = solve(H, g, assume_a = "pos", overwrite_b = True)
    #dx = spsolve(csr_matrix(H), -g)
    from sksparse.cholmod import cholesky
    

    # Perform Cholesky factorization
    H.flat[::H.shape[0]+1] +=  0.0001
    factor = cholesky(csc_matrix(H))

    dx = factor.solve_A(-g)
    end = time.time()
    time_diff = end - start
    print("solve time: %f"%time_diff)
    exit(0)
    """
    
    

    print("Load dataset...")
    loader = BALLoader()
    filename = "data/bal/problem-49-7776-pre.txt"
    if loader.load_file(filename):
        print("File loaded successfully!")
        print("Number of cameras:", len(loader.cameras))
        print("Number of points:", len(loader.points))
        print("Number of observation:", len(loader.observations))
    else:
        print("Error loading file.")
    #fig = plt.figure('test')
    #plt.axis('equal')
    #ax = fig.add_subplot(projection='3d')

    print("Undistort points...")
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        obs.u_undist = undistort_point(obs.u, cam.K, cam.dist_coeffs)

    graph = GraphSolver(use_sparse=True)

    print("Add camera vertex...")
    for i, cam in enumerate(loader.cameras):
        T = makeT(cam.R, cam.t)
        if(i == 0):
            graph.add_vertex(CameraVertex(T), is_constant=True)
        else:
            graph.add_vertex(CameraVertex(T))  # add vertex to graph
            graph.add_edge(Pose3dEdge([i], T, np.eye(6) * 1)) 

    print("Add point vertex...")
    for pw in loader.points:
        graph.add_vertex(PointVertex(pw))

    print("Undistort points...")
    camera_size = len(loader.cameras)
    kernel = HuberKernel(np.sqrt(5))
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        graph.add_edge(ReprojEdge(
            [obs.camera_id, camera_size + obs.point_id],
            [obs.u_undist, cam.K],
            np.eye(2), kernel)) 

    print("solve...")
    graph.solve(True, min_score_change=0.5)
    cameras = []
    points = []
    for i, v in enumerate(graph.vertices):
        if (i < len(loader.cameras)):
            cameras.append(m2p(v.x))
        else:
            points.append(v.x)

    cameras = np.array(cameras)
    points = np.array(points)

    np.save("cameras.npy", cameras)
    np.save("points.npy", points)


    #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    #plt.show()
    


