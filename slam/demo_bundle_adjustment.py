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
        Twc = vertices[self.link[0]].x
        pw = vertices[self.link[1]].x
        u, k = self.z
        r, JTwc, Jpw = reproj0(Twc, pw, u, k, True)
        return r, [JTwc, Jpw]
    

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
    for obs in loader.observations:
        cam = loader.cameras[obs.camera_id]
        graph.add_edge(ReprojEdge([obs.camera_id, camera_size + obs.point_id], [obs.u_undist, cam.K])) 

    print("solve...")

    graph.solve(True)
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    #plt.show()


