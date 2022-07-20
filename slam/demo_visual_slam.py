import numpy as np
#from graph_solver import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *

import yaml

frames = []
points = {}
n = 10
for idx in range(n):
    with open('data/slam/F%04d.yaml'%idx) as file:
        node = yaml.safe_load(file)
        pts = np.array(node['points']['data']).reshape(node['points']['num'],-1)
        imus = np.array(node['imu']['data']).reshape(node['imu']['num'],-1)
        frames.append({'stamp':node['stamp'],'points': dict(zip(pts[:,0].astype(np.int), pts[:,1:])),'imu':imus})
print(frames[0])

for frame in frames:
    for p in frame['points']:
        print(p)
        points.update({p: np.array([0,0,0.])})
print(points)