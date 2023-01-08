from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  

import numpy as np
#from stl import mesh

from pathlib import Path

def cube():
    vertexes = np.array([[1, 0, 0], #0
                     [0, 0, 0], #1
                     [0, 1, 0], #2
                     [0, 0, 1], #3
                     [1, 1, 0], #4
                     [1, 1, 1], #5
                     [0, 1, 1], #6
                     [1, 0, 1]])#7
    faces = np.array([[1,0,7], [1,3,7],
                      [1,2,4], [1,0,4],
                      [1,2,6], [1,3,6],
                      [0,4,5], [0,7,5],
                      [2,4,5], [2,6,5],
                      [3,6,5], [3,7,5]])
    colors = np.array([[1,0,0,1],[1,0,0,1],
                       [0,1,0,1],[0,1,0,1],
                       [0,0,1,1],[0,0,1,1],
                       [1,1,0,1],[1,1,0,1],
                       [0,1,1,1],[0,1,1,1],
                       [1,0,1,1],[1,0,1,1]])
    return gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors,
                     drawEdges=True, edgeColor=(0, 0, 0, 1))



class MyWindow(QMainWindow):
    def __init__(self):
        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0
        super(MyWindow, self).__init__()
        self.setGeometry(0, 0, 700, 900) 
        self.initUI()


    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        layout.addWidget(self.viewer, 1)

        self.label_roll = QLabel()
        self.label_roll.setText("roll:  %0.2f dgree"%self.rotX)
        self.label_pitch = QLabel()
        self.label_pitch.setText("pitch: %0.2f dgree"%self.rotY)
        self.label_yaw = QLabel()
        self.label_yaw.setText("yaw:   %0.2f dgree"%self.rotZ)

        sliderX = QSlider(QtCore.Qt.Horizontal)
        sliderX.setMinimum(0)
        sliderX.setMaximum(36000)
        sliderX.valueChanged.connect(lambda val: self.setRotX(val))

        sliderY = QSlider(QtCore.Qt.Horizontal)
        sliderY.setMinimum(0)
        sliderY.setMaximum(36000)
        sliderY.valueChanged.connect(lambda val: self.setRotY(val))

        sliderZ = QSlider(QtCore.Qt.Horizontal)
        sliderZ.setMinimum(0)
        sliderZ.setMaximum(36000)
        sliderZ.valueChanged.connect(lambda val: self.setRotZ(val))
        layout.addWidget(self.label_roll)
        layout.addWidget(sliderX)
        layout.addWidget(self.label_pitch)
        layout.addWidget(sliderY)
        layout.addWidget(self.label_yaw)
        layout.addWidget(sliderZ)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.update)
    

        self.viewer.setWindowTitle('Euler rotation')
        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(5, 5)
        self.viewer.addItem(g)

        self.cube = cube()
        self.viewer.addItem(self.cube)
        timer.start()

    def setRotX(self, val):
        self.rotX = float(val/100.)
        self.label_roll.setText("roll:  %0.2f dgree"%self.rotX)

    def setRotY(self, val):
        self.rotY = float(val/100.)
        self.label_roll.setText("pitch: %0.2f dgree"%self.rotY)

    def setRotZ(self, val):
        self.rotZ = float(val/100.)
        self.label_roll.setText("yaw:   %0.2f dgree"%self.rotZ)

    def update(self):
        self.cube.resetTransform()
        self.cube.rotate(self.rotX, 1, 0, 0)
        self.cube.rotate(self.rotY, 0, 1, 0)
        self.cube.rotate(self.rotZ, 0, 0, 1)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
