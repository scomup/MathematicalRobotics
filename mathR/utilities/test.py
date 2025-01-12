import numpy as np
import matplotlib.pyplot as plt
from mathR.utilities import *
from mathR.gauss_newton_method.gauss_newton import *
from mathR.robot_geometry.geometry_plot import *


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
    for i in range(n):
        for j in range(0, n-i-1):
            if clockwise_check(points[j], points[j+1]) < 0:
                tmp = np.copy(points[j])
                points[j] = points[j+1]
                points[j+1] = tmp
    return points


def cross_cube(plane_point, plane_normal, cube_p, cube_size):
    points = []
    cube_vertices, cube_edges = cube(cube_p, cube_size)
    for edge in cube_edges:
        p = find_cross_point(plane_point, plane_normal, cube_vertices[edge[0]], cube_vertices[edge[1]])
        if p is not None:
            points.append(p)
    return np.array(points)


def cube(cube_p, cube_size):
    # Calculate the remaining vertices of the cube
    p1 = np.array([cube_p[0], cube_p[1], cube_p[2]])
    p2 = np.array([cube_p[0], cube_p[1]+cube_size[1], cube_p[2]])
    p3 = np.array([cube_p[0], cube_p[1]+cube_size[1], cube_p[2]+cube_size[2]])
    p4 = np.array([cube_p[0], cube_p[1], cube_p[2]+cube_size[2]])
    p5 = np.array([cube_p[0]+cube_size[2], cube_p[1], cube_p[2]])
    p6 = np.array([cube_p[0]+cube_size[2], cube_p[1]+cube_size[1], cube_p[2]])
    p7 = np.array([cube_p[0]+cube_size[2], cube_p[1]+cube_size[1], cube_p[2]+cube_size[2]])
    p8 = np.array([cube_p[0]+cube_size[2], cube_p[1], cube_p[2]+cube_size[2]])

    # Define the vertices of the cube
    cube_vertices = np.array([p1, p2, p3, p4, p5, p6, p7, p8])
    cube_edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4],
                  [0, 4], [1, 5], [2, 6], [3, 7]]
    return cube_vertices, cube_edges


def draw_cube(ax, cube_p, cube_size):
    cube_vertices, cube_edges = cube(cube_p, cube_size)
    # Plot the cube
    for edge in cube_edges:
        ax.plot(cube_vertices[edge, 0], cube_vertices[edge, 1], cube_vertices[edge, 2], 'r')


def draw_lines(ax, points):
        n = points.shape[0]
        for i in range(n):
            idx = [i, (i + 1) % n]
            ax.plot(points[idx, 0], points[idx, 1], points[idx, 2], 'b')


if __name__ == '__main__':
    fig = plt.figure("plane", figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    p = np.array([0.2, 0.3, 0.1])
    norm = np.array([0.2, 0.1, 0.9])
    norm = norm/np.linalg.norm(norm)
    plane = np.zeros(4)
    plane[0:3] = norm
    cube_p = np.array([-1, -1, -1])
    cube_size = np.array([2, 2, 2.])
    draw_arrow(ax, p, norm)
    draw_plane2(ax, p, get_R_by_norm(norm), size=[2, 2])
    draw_cube(ax, cube_p, cube_size)
    points = cross_cube(p, norm, cube_p, cube_size)
    points = order_points_clockwise(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black')
    draw_lines(ax, points)
    set_axes_equal(ax)

    ax.legend()
    plt.show()
