import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from pypcd import pypcd
# pc1 = pypcd.PointCloud.from_path('/Users/liuyang/workspace/MathematicalRobotics/robot_geometry/2.pcd')

pcd1 = o3d.io.read_point_cloud("/Users/liuyang/workspace/MathematicalRobotics/robot_geometry/1.pcd", format='pcd')
pcd2 = o3d.io.read_point_cloud("/Users/liuyang/workspace/MathematicalRobotics/robot_geometry/2.pcd", format='pcd')
pcd1.paint_uniform_color([0.1, 0.9, 0.1])
pcd2.paint_uniform_color([0.9, 0.1, 0.1])
pc = np.asarray(pcd1.points)

a = []
a2 = []

coln = 360
rown = 16
img = np.zeros([rown, coln])
for p in pc:
    range = np.linalg.norm(p)
    l = p/p[2]
    col_angle = np.arctan2(l[1], l[0])+np.pi/2
    row_angle = np.arcsin(p[2]/range) + 15/180*np.pi
    col_id = int(col_angle / np.pi * coln)
    row_id = int(row_angle / (30/180*np.pi) * rown)
    if (row_id >= rown or col_id >= coln):
        continue
    img[row_id, col_id] = range

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
axes.imshow(img, aspect='auto')
plt.show()

# print("Load a ply point cloud, print it, and render it")
# print(pcd)
# print(np.asarray(pcd.points))
# o3d.visualization.draw_geometries([pcd1, pcd2])
