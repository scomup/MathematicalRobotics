import numpy as np
from math_tools import *
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
    axes = fig.gca()
    plot_pose2_on_axes(axes,
                       pose,
                       axis_length=axis_length)

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])

    return fig