
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graph_optimization.graph_solver import *
from utilities.math_tools import *
import numpy as np
import matplotlib.pyplot as plt
from graph_optimization.plot_pose import *

def set_axes(fignum: int, ) -> None:
    fig = plt.figure(fignum)
    if not fig.axes:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.axes[0]

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])

def euler2M(euler):
    """
    xyz order
    """
    pitch, roll, yaw = euler
    c1 = np.cos(pitch)
    s1 = np.sin(pitch)
    c2 = np.cos(roll)
    s2 = np.sin(roll)
    c3 = np.cos(yaw)
    s3 = np.sin(yaw)
    matrix=np.array([[c2*c3, -c2*s3, s2],
                     [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                     [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    return matrix


if __name__ == '__main__':
    x = np.array([0,0,0,np.pi/8, np.pi/12, -np.pi/12])
    M1 = expSE3(x)
    M2 = np.eye(4)
    M2[0:3,0:3] =  euler2M(x[3:6])
    M3 = np.eye(4)
    n = 100
    for i in range(n):
        #M3[0:3,0:3] = M3[0:3,0:3].dot(euler2M(x[3:6]/n))
        M3[0:3,0:3] = euler2M(x[3:6]/n).dot(M3[0:3,0:3])
        #M3[0:3,0:3] = (np.eye(3) + skew(x[3:6]/n)).dot(M3[0:3,0:3])
    print(M1)
    print(M2)
    print(M3)
    plot_pose3("org", np.eye(4), 0.5)
    plot_pose3("so3", M1, 0.5)
    plot_pose3("euler", M2, 0.5)
    plot_pose3("exp", M3, 0.5)
    set_axes("org")
    set_axes("so3")
    set_axes("euler")
    set_axes("exp")
    plt.show()


