from math_tools import *


def draw_plane(ax, plane, center=np.array([0, 0]), size=[5, 5]):
    xx, yy = np.meshgrid(np.arange(center[0] - size[0], center[0] + size[0]),
                         np.arange(center[1] - size[1], center[1] + size[1]))
    a, b, c, d = plane
    z = -(a*xx+b*yy+d)/c
    ax.plot_surface(xx, yy, z, alpha=0.5)
