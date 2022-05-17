import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from matplotlib import patches
import matplotlib.pyplot as plt


def plot_pose2_on_axes(axes,
                       pose,
                       axis_length: float = 0.1):

    gRp, origin = makeRt(v2m(pose))

    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'r-')

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    axes.plot(line[:, 0], line[:, 1], 'g-')

    e1 = patches.Circle(xy=origin, radius=axis_length, fill=False)

    axes.add_patch(e1)


def plot_pose2(
        fignum: int,
        pose,
        axis_length: float = 0.1,
        axis_labels=("X axis", "Y axis", "Z axis")):
    # get figure object
    fig = plt.figure(fignum)
    plt.axis('equal')
    axes = fig.gca()
    plot_pose2_on_axes(axes,
                       pose,
                       axis_length=axis_length)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])

    return fig


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utilities.math_tools import *

    pose = []
    cur_pose = np.array([0,0,0])
    odom = np.array([0.2, 0, 0.5])
    for i in range(12):
        pose.append(cur_pose)
        cur_pose = m2v(v2m(cur_pose).dot(v2m(odom)))
        plot_pose2(0, cur_pose, 0.05)
    plt.axis('equal')
    plt.show()

