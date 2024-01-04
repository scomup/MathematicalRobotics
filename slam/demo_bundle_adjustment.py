from load_bal_datasets import BALLoader
from reprojection import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


class CameraVertex(BaseVertex):
    def __init__(self, x):
        super().__init__(x, 6)

    def update(self, dx):
        self.x = self.x @ p2m(dx)


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
        """
        R, t = makeRt(Tcw)
        pc = R @ pw + t
        fx = K[0, 0]
        fy = K[1, 1]
        x, y, z = pc
        z_2 = z * z
        tmp = np.zeros([2, 3])
        tmp[0, 0] = fx
        tmp[0, 1] = 0
        tmp[0, 2] = -x / z * fx
        tmp[1, 0] = 0
        tmp[1, 1] = fy
        tmp[1, 2] = -y / z * fy      
        Jxi = -1. / z * tmp @ R
        Jxj = np.zeros([2, 6])
        Jxj[0, 3] = x * y / z_2 * fx
        Jxj[0, 4] = -(1 + (x * x / z_2)) * fx
        Jxj[0, 5] = y / z * fx
        Jxj[0, 0] = -1. / z * fx
        Jxj[0, 1] = 0
        Jxj[0, 2] = x / z_2 * fx

        Jxj[1, 3] = (1 + y * y / z_2) * fy
        Jxj[1, 4] = -x * y / z_2 * fy
        Jxj[1, 5] = -x / z * fy
        Jxj[1, 0] = 0
        Jxj[1, 1] = -1. / z * fy
        Jxj[1, 2] = y / z_2 * fy
        """

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


if __name__ == '__main__':   

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

    print("Add point vertex...")
    for pw in loader.points:
        graph.add_vertex(PointVertex(pw))

    print("Undistort points...")
    camera_size = len(loader.cameras)
    kernel = PseudoHuberKernel(2)
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        graph.add_edge(ReprojEdge(
            [obs.camera_id, camera_size + obs.point_id],
            [obs.u_undist, cam.K],
            np.eye(2), kernel)) 

    print("solve...")

    graph.solve(True)
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    #plt.show()


