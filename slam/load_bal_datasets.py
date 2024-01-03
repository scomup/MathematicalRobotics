import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


class Camera:
    def __init__(self):
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.focal = 0.0
        self.k1 = 0.0
        self.k2 = 0.0


class Observation:
    def __init__(self):
        self.camera_idx = 0
        self.point_idx = 0
        self.pt = np.zeros(2)


class BALLoader:
    """
    The BAL Data Format is described at the following URL.
    https://grail.cs.washington.edu/projects/bal/
    """
    def __init__(self):
        self.observations = []
        self.cameras = []
        self.points = []

    def load_file(self, filename: str) -> bool:
        try:
            with open(filename, 'r') as f:
                # read BAL header
                num_cameras, num_points, num_observations = map(int, f.readline().split())
                
                # read observations
                self.observations = [Observation() for _ in range(num_observations)]
                for i in range(num_observations):
                    obs = self.observations[i]
                    obs.camera_idx, obs.point_idx, obs.pt[0], obs.pt[1] = map(float, f.readline().split())

                # read cameras
                self.cameras = [Camera() for _ in range(num_cameras)]
                for i in range(num_cameras):
                    rot_vec = np.array([float(f.readline()), float(f.readline()), float(f.readline())])
                    translation = np.array([float(f.readline()), float(f.readline()), float(f.readline())])
                    self.cameras[i].R = expSO3(rot_vec)
                    self.cameras[i].t = translation
                    self.cameras[i].focal = float(f.readline())
                    self.cameras[i].k1 = float(f.readline())
                    self.cameras[i].k2 = float(f.readline())

                # read points
                for i in range(num_points):
                    self.points.append(np.array([float(f.readline()), float(f.readline()), float(f.readline())]))
            return self.observations

        except (IOError, ValueError, IndexError):
            return False


# Example usage:
loader = BALLoader()
filename = "data/bal/problem-49-7776-pre.txt"
if loader.load_file(filename):
    print("File loaded successfully!")
    print("Number of cameras:", len(loader.cameras))
    print("Number of points:", len(loader.points))
    print("Number of observation:", len(loader.observations))
else:
    print("Error loading file.")
