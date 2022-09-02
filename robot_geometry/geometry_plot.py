import numpy as np

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from graph_optimization.plot_pose import set_axes_equal


def set_axes_equal(ax) -> None:

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])



class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        from mpl_toolkits.mplot3d.proj3d import proj_transform
        #from mpl_toolkits.mplot3d.axes3d import Axes3D
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def draw_point(ax, p, name='none'):
    ax.scatter(p[0], p[1], p[2], s=30, marker='o', color='green')
    ax.annotate3D(name, p, xytext=(3, 3), textcoords='offset points')

def draw_arrow(ax, org, direction, name='none'):
    ax.arrow3D(*org, *direction, mutation_scale=20, arrowstyle="-|>", linestyle='dashed')
    center = org + direction * 0.5
    ax.annotate3D(name, center, xytext=(3, 3), textcoords='offset points')

def draw_line(ax, org, direction, name='none', length=5):
    a = org + direction * 0.5 * length
    b = org - direction * 0.5 * length
    ax.plot([a[0], b[0]],[a[1], b[1]],[a[2], b[2]])
    ax.annotate3D(name, org, xytext=(3, 3), textcoords='offset points')

def draw_plane(ax, plane, center=np.array([0,0]), size=[5,5]):
    xx, yy = np.meshgrid(np.arange(center[0]-size[0],center[0]+size[0]), np.arange(center[1]-size[1],center[1]+size[1]))
    a, b, c, d = plane
    z = -(a*xx+b*yy+d)/c
    ax.plot_surface(xx, yy, z, alpha=0.5)


if __name__ == '__main__':
    fig = plt.figure("line")
    ax = fig.add_subplot(111, projection='3d')
    a = np.array([-1,1,0])
    b = np.array([1,2,3])
    p = np.array([3,0,1])
    ab = b-a
    pa = a-p
    pb = b-p
    pa = a - p
    pb = b - p
    ab = b - a
    pm = np.cross(pa, pb)
    ab_norm = np.linalg.norm(ab)
    pm_norm = np.linalg.norm(pm)
    d = pm_norm/ab_norm
    p2ab = np.cross(-pm, ab)/(ab_norm*pm_norm)*d
    draw_point(ax, a, 'a')
    draw_point(ax, b, 'b')
    draw_point(ax, p, 'p')
    draw_arrow(ax, a, ab, 'ab')
    draw_arrow(ax, p, pa, 'pa')
    draw_arrow(ax, p, pb, 'pb')
    draw_arrow(ax, p, p2ab, 'p2ab')
    set_axes_equal(ax)

    fig = plt.figure("plane")
    ax1 = fig.add_subplot(111, projection='3d')
    draw_point(ax1, p, 'p')
    plane = np.array([0,0,1,1])
    draw_plane(ax1, plane, p)
    set_axes_equal(ax1)

    plt.show()
