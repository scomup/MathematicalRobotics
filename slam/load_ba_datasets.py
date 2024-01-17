import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


class Camera:
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.K = np.eye(3)  # camera matrix
        self.dist_coeffs = np.zeros(2)


class Observation:
    def __init__(self):
        self.camera_id = 0
        self.point_id = 0
        self.u = np.zeros(2)  # point in image
        self.u_undist = np.zeros(2)


class BALLoader:
    """
    The BAL Data Format is described at the following URL.
    https://grail.cs.washington.edu/projects/bal/
    """
    def __init__(self):
        self.observations = []
        self.cameras = []
        self.points = []
        self.colors = None

    def load_file(self, filename: str) -> bool:
        try:
            with open(filename, 'r') as f:
                # read BAL header
                num_cameras, num_points, num_observations = map(int, f.readline().split())

                # read observations
                for i in range(num_observations):
                    obs = Observation()
                    obs.camera_id, obs.point_id, obs.u[0], obs.u[1] = map(float, f.readline().split())
                    obs.camera_id = int(obs.camera_id)
                    obs.point_id = int(obs.point_id)
                    obs.u = -obs.u
                    obs.u_undist = obs.u
                    self.observations.append(obs)

                # read cameras
                for i in range(num_cameras):
                    rot_vec = np.array([float(f.readline()), float(f.readline()), float(f.readline())])
                    translation = np.array([float(f.readline()), float(f.readline()), float(f.readline())])
                    cam = Camera()
                    Tcw = makeT(expSO3(rot_vec), translation)
                    Twc = np.linalg.inv(Tcw)
                    cam.R = Twc[0:3, 0:3]
                    cam.t = Twc[0:3, 3]
                    focal = float(f.readline())
                    cam.dist_coeffs[0] = float(f.readline())  # k1
                    cam.dist_coeffs[1] = float(f.readline())  # k2
                    cam.K[0, 0] = focal
                    cam.K[1, 1] = focal
                    self.cameras.append(cam)

                # read points
                for i in range(num_points):
                    self.points.append(np.array([float(f.readline()), float(f.readline()), float(f.readline())]))

                self.points = np.array(self.points)
                return True

        except (IOError, ValueError, IndexError):
            return False

if __name__ == '__main__':
    # Example usage:
    loader = BALLoader()
    filename = "data/ba/problem-49-7776-pre.txt"
    if loader.load_file(filename):
        print("File loaded successfully!")
        print("Number of cameras:", len(loader.cameras))
        print("Number of points:", len(loader.points))
        print("Number of observation:", len(loader.observations))
    else:
        print("Error loading file.")
