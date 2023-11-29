import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.plot_tools import draw_cube, cube, draw_lines
from guass_newton_method.guass_newton import *
from robot_geometry.geometry_plot import *


def find_cross_point(plane_point, plane_normal, a, b):
    dir = b - a
    t = np.dot(plane_normal, plane_point - a) / np.dot(plane_normal, dir)
    if t >= 0 and t <= 1:
        cross_point = a + t * dir
        return cross_point
    return None


def order_points_clockwise(points):
    centroid = np.mean(points, axis=0)
    vec1 = points[0] - centroid
    vec2 = points[1] - centroid
    normal = np.cross(vec1, vec2)
    normal /= np.linalg.norm(normal)

    def clockwise_check(a, b):
        cross_product = np.cross(a - centroid, b - centroid)
        dot_product = np.dot(normal, cross_product)
        return dot_product

    n = points.shape[0]
    for i in range(n-1):
        min_dist = np.inf
        idx = i+1
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if clockwise_check(points[i], points[j]) > 0 and dist < min_dist:
                min_dist = dist
                idx = j
        if idx == i+1:
            continue
        tmp = np.copy(points[i+1])
        points[i+1] = points[idx]
        points[idx] = tmp
    return points


def cross_cube(plane_point, plane_normal, cube_p, cube_size):
    points = []
    cube_vertices, cube_edges = cube(cube_p, cube_size)
    for edge in cube_edges:
        p = find_cross_point(plane_point, plane_normal, cube_vertices[edge[0]], cube_vertices[edge[1]])
        if p is not None:
            points.append(p)
    return np.array(points)


if __name__ == '__main__':
    fig = plt.figure("plane", figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    p = np.array([-2, -0.1, 0.3])
    norm = np.array([2, -0.5, 0.5])
    norm = norm/np.linalg.norm(norm)

    for i in range(40):
        # plane = np.zeros(4)
        # plane[0:3] = norm
        ax.clear()
        cube_p = np.array([-1, -1, -1])
        cube_size = np.array([2, 2, 2.])
        # draw_arrow(ax, p, norm)
        p[0] += 0.1
        draw_plane2(ax, p, get_R_by_norm(norm), size=[3, 3])
        draw_cube(ax, np.eye(4), cube_p, cube_size, color='r')
        points = cross_cube(p, norm, cube_p, cube_size)
        if(points.shape[0] <= 2):
            continue
        points = order_points_clockwise(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black')
        draw_lines(ax, points)
        set_axes_equal(ax)
        # ax.legend()
        plt.pause(0.1)
