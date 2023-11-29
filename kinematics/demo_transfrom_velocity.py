from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QApplication, QPushButton
from OpenGL.GL import *
from PyQt5 import QtCore, QtGui
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from utilities.gl_objects import *
from transfrom_velocity import transformVelocity3D


class GLTextViewWidget(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colorA = QtGui.QColor.fromRgbF(0.8, 0.2, 0.2, 1)
        self.colorB = QtGui.QColor.fromRgbF(0.2, 0.8, 0.2, 1)
        self.textA = 'Velocities of A'
        self.textB = 'Velocities of B'

    def paintGL(self):
        super().paintGL()
        painter = QtGui.QPainter(self)
        font = QtGui.QFont('Helvetica', 15)
        painter.setFont(font)
        color = QtGui.QColor.fromRgbF(1, 1, 1, 1)  # white color
        painter.setPen(self.colorA)
        Y = 600
        painter.drawText(100, Y+50, self.textA)
        painter.setPen(self.colorB)
        painter.drawText(100, Y+80, self.textB)
        painter.end()

    def set_textA(self, text):
        self.textA = text
        self.update()

    def set_textB(self, text):
        self.textB = text
        self.update()


class Gui3d(QMainWindow):
    def __init__(self, robot_arm, traj_A, traj_B, va, dt):
        self.time = 0.0
        self.robot_arm = robot_arm
        self.traj_A = traj_A
        self.traj_B = traj_B
        self.pre_A = np.eye(4)
        self.pre_B = np.eye(4)
        self.pre_time = 0.
        self.va = va
        self.dt = dt
        self.T = np.eye(4)

        super(Gui3d, self).__init__()
        self.setGeometry(0, 0, 700, 900)
        self.initUI()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = GLTextViewWidget()
        layout.addWidget(self.viewer, 1)

        self.label_text = QLabel()
        self.label_text.setText("time: %3.3f" % (self.time))

        slider_x = QSlider(QtCore.Qt.Horizontal)
        slider_x.setMinimum(0)
        slider_x.setMaximum(2000)
        slider_x.valueChanged.connect(lambda val: self.setX(val))

        button = QPushButton("start")
        self.start = True
        button.clicked.connect(self.button_cb)

        layout.addWidget(self.label_text)
        layout.addWidget(slider_x)
        layout.addWidget(button)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.update)

        self.viewer.setWindowTitle('Euler rotation')
        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(5, 5)
        self.viewer.addItem(g)

        self.viewer.addItem(self.robot_arm)
        self.viewer.addItem(self.traj_A)
        self.viewer.addItem(self.traj_B)
        timer.start()

    def setX(self, vel):
        self.time = float(vel/100.)
        self.label_text.setText("time: %3.3f " % (self.time))

    def button_cb(self):
        self.start = not self.start

    def update(self):
        if (self.start):
            self.time += self.dt
        time = self.time
        dt = time - self.pre_time

        self.T = self.T @ expSE3(self.va * dt)
        self.robot_arm.setTransform(self.T)

        A = self.robot_arm.getA()
        B = self.robot_arm.getB()
        self.traj_A.addPoints(A[0:3, 3])
        self.traj_B.addPoints(B[0:3, 3])
        deltaB = np.linalg.inv(self.pre_B) @ B
        if (dt != 0):
            Rb, tb = makeRt(deltaB)
            omega_b = logSO3(Rb)/dt
            v_b = tb/dt
            vb = np.concatenate((v_b, omega_b))
            self.viewer.set_textA("Velocities of A: [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f]" %
                                  (va[0], va[1], va[2], va[3], va[4], va[5]))
            self.viewer.set_textB("Velocities of B: [%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f]" %
                                  (vb[0], vb[1], vb[2], vb[3], vb[4], vb[5]))
        self.pre_A = A
        self.pre_B = B
        self.pre_time = time
        self.label_text.setText("time: %3.3f " % (self.time))
        if (self.time < 0.1):
            self.traj_A.clear()
            self.traj_B.clear()
        self.viewer.update()


if __name__ == '__main__':

    # try different values at here
    Tba = expSE3(np.array([2, 5, 10, 1., 0.3, 0.1]))
    va = np.array([5, 0.8, 1, 0, 0.0, 2])
    dt = 0.1  #

    vb = transformVelocity3D(Tba, va)
    print("The Velocities of B is:", vb)
    app = QApplication([])
    axis = GLAxisItem(size=[10, 10, 10], width=100)
    colorA = np.array([1, 0, 0, 1])
    colorB = np.array([0, 1, 0, 1])
    arm = GLRobotARMItem(Tba, colorA, colorB)
    trajA = GLTrajItem(width=5, color=np.append(colorA[0:3], 0.3))
    trajB = GLTrajItem(width=5, color=np.append(colorB[0:3], 0.3))
    window = Gui3d(arm, trajA, trajB, va, dt)
    window.show()

    app.exec_()
