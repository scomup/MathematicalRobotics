from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np


def euler2mat(euler, order='xyz'):
    roll, pitch, yaw = euler
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    R = None
    # print(order)
    if (order == 'xyz'):
        R = Rz.dot(Ry.dot(Rx))
        # r = Rotation.from_euler("xyz", euler, degrees=False)
        # print(R)
        # print("-------")
        # print(r.as_matrix())
        # print("=======")
    elif (order == 'yzx'):
        R = Rx.dot(Rz.dot(Ry))
    elif (order == 'zxy'):
        R = Ry.dot(Rx.dot(Rz))
    elif (order == 'xzy'):
        R = Ry.dot(Rz.dot(Rx))
    elif (order == 'zyx'):
        R = Rx.dot(Ry.dot(Rz))
    elif (order == 'yxz'):
        R = Rz.dot(Rx.dot(Ry))
    return R


class GLAxisItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=[1, 1, 1], width=100, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        x, y, z = size
        self.width = width
        self.axis_x = np.array([x, 0, 0, 1])
        self.axis_y = np.array([0, y, 0, 1])
        self.axis_z = np.array([0, 0, z, 1])
        self.orig = np.array([0, 0, 0, 1])
        self.setGLOptions(glOptions)
        self.T = np.eye(4)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        axis_x = self.T.dot(self.axis_x)
        axis_y = self.T.dot(self.axis_y)
        axis_z = self.T.dot(self.axis_z)
        orig = self.T.dot(self.orig)
        self.setupGLState()
        glLineWidth(self.width)
        glBegin(GL_LINES)
        glColor4f(0, 0, 1, 1)  # z is blue
        glVertex3f(orig[0], orig[1], orig[2])
        glVertex3f(axis_z[0], axis_z[1], axis_z[2])
        glColor4f(0, 1, 0, 1)  # y is green
        glVertex3f(orig[0], orig[1], orig[2])
        glVertex3f(axis_y[0], axis_y[1], axis_y[2])
        glColor4f(1, 0, 0, 1)  # x is red
        glVertex3f(orig[0], orig[1], orig[2])
        glVertex3f(axis_x[0], axis_x[1], axis_x[2])
        glEnd()

class GLCameraFrameItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=1, width=1, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.width = width
        self.setGLOptions(glOptions)
        self.T = np.eye(4)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        self.setupGLState()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLineWidth(self.width)
        glBegin(GL_LINES)
        hsize = self.size / 2

        # Draw the square base of the pyramid
        #glMultMatrixf(self.T.T)

        frame_points =  np.array([[-hsize, -hsize, 0, 1],
                                [hsize, -hsize, 0, 1],
                                [hsize, hsize, 0, 1],
                                [-hsize, hsize, 0, 1],
                                [0, 0, hsize, 1]])
        frame_points = (self.T @ frame_points.T).T[:,0:3]
        glColor4f(0, 0, 1, 1)
        glVertex3f(*frame_points[0])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[3])
        glVertex3f(*frame_points[3])
        glVertex3f(*frame_points[0])
        # Draw the four lines representing the triangular sides of the pyramid
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[0])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[1])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[2])
        glVertex3f(*frame_points[4])
        glVertex3f(*frame_points[3])
        glEnd()

class Gui3d(QMainWindow):
    def __init__(self, static_obj):
        self.static_obj = static_obj
        super(Gui3d, self).__init__()
        self.setGeometry(0, 0, 1200, 800)
        self.initUI()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 1)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.update)

        self.viewer.setWindowTitle('Euler rotation')
        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(5, 5)
        self.viewer.addItem(g)

        for obj in self.static_obj:
            self.viewer.addItem(obj)

        timer.start()

    def update(self):
        self.viewer.update()


if __name__ == '__main__':
    app = QApplication([])
    axis = GLAxisItem(size=[1, 1, 1], width=2)
    cam = GLCameraFrameItem(size=1, width=1)
    window = Gui3d(static_obj=[cam, axis])
    window.show()
    app.exec_()
