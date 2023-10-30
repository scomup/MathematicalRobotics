from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication, QPushButton
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation   
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *
from utilities.gl_objects import *

pi = np.array([1, 0, 0.3,    0, 0.1, 0])
pj = np.array([0, 1, 0.3,    -0.1, 0, 0])


class Gui3d(QMainWindow):
    def __init__(self, static_obj=[], dynamic_obj=[], trajs=None):
        self.x = 0.0
        self.y = 0.0
        self.static_obj = static_obj
        self.dynamic_obj = dynamic_obj
        self.trajs = trajs
        super(Gui3d, self).__init__()
        self.setGeometry(0, 0, 700, 900)
        self.initUI()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 1)

        self.label_text = QLabel()
        self.label_text.setText("x: %3.2f y: %3.2f" % (self.x, self.y))

        slider_x = QSlider(QtCore.Qt.Horizontal)
        slider_x.setMinimum(0)
        slider_x.setMaximum(2000)
        slider_x.valueChanged.connect(lambda val: self.setX(val))

        # slider_y = QSlider(QtCore.Qt.Horizontal)
        # slider_y.setMinimum(0)
        # slider_y.setMaximum(2000)
        # slider_y.valueChanged.connect(lambda val: self.setY(val))

        button = QPushButton("start")
        self.start = False
        button.clicked.connect(self.button_cb)

        layout.addWidget(self.label_text)
        layout.addWidget(slider_x)
        layout.addWidget(button)
        # layout.addWidget(slider_y)

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
        for obj in self.dynamic_obj:
            self.viewer.addItem(obj)
        for obj in self.trajs:
            self.viewer.addItem(obj)

        timer.start()

    def setX(self, vel):
        self.x = float(vel/100.)
        self.label_text.setText("x: %3.2f y: %3.2f" % (self.x, self.y))

    def button_cb(self):
        self.start = not self.start

    def setY(self, vel):
        self.y = float(vel/100.)
        self.label_text.setText("x: %3.2f y: %3.2f" % (self.x, self.y))

    def update(self):
        x = (0.5*pi+0.3*pj)*self.x
        T = expSE3(x)
        for obj in self.dynamic_obj:
            obj.setTransform(T)

        self.trajs[0].addPoints(T[0:3, 3])
        self.trajs[1].addPoints(x[0:3])
        if (self.start):
            self.x += 0.1

        self.viewer.update()

if __name__ == '__main__':
    app = QApplication([])
    axis = GLAxisItem(size=[10, 10, 10], width=100)
    traj1 = GLTrajItem(color=[1, 0, 0, 1])
    traj2 = GLTrajItem(color=[0, 1, 1, 1])
    surf = GLSurfItem(color=[0, 1, 0, 0.2], pi=pi, pj=pj)
    plant = GLPlantItem(color=[0, 1, 1, 0.1], pi=pi, pj=pj)
    cube = create_cube(1)
    window = Gui3d(static_obj=[surf, plant], dynamic_obj=[cube], trajs=[traj1, traj2])
    window.show()
    app.exec_()
