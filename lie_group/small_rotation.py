from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
#from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *

import numpy as np

class GLArrowItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size = 1., width = 100, color = [1,0,0,1], glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.width = width
        self.color = color
        self.o = np.array([0,0,0,1])
        self.a = np.array([size,0,0,1])
        self.b = np.array([0.9*size,size*0.1,0,1])
        self.setGLOptions(glOptions)
        self.T = np.eye(4)

    def setTransform(self, T):
        #print(T)
        self.T = T

    def paint(self):
        o = self.T.dot(self.o) 
        a = self.T.dot(self.a) 
        b = self.T.dot(self.b) 
        glLineWidth(self.width)
        glBegin( GL_LINES )
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])  # z is blue
        glVertex3f(o[0], o[1], o[2])
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])
        glEnd()

def approx_euler2mat(euler, order='xyz'):
    euler = np.array(euler)
    return skew(euler) + np.eye(3)


    
def euler2mat(euler, order='xyz'):
    roll, pitch, yaw = euler
    Rx =  np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(roll),-np.sin(roll)],
                   [ 0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0           , 1, 0           ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0 ],
                   [ np.sin(yaw), np.cos(yaw) , 0 ],
                   [ 0           , 0            , 1 ]])
    R = None
    if(order=='xyz'):
        R = Rz.dot(Ry.dot(Rx))
    elif(order=='yzx'):
        R = Rx.dot(Rz.dot(Ry))
    elif(order=='zxy'):
        R = Ry.dot(Rx.dot(Rz))
    elif(order=='xzy'):
        R = Ry.dot(Rz.dot(Rx))
    elif(order=='zyx'):
        R = Rx.dot(Ry.dot(Rz))
    elif(order=='yxz'):
        R = Rz.dot(Rx.dot(Ry))
    return R

def ball(size = 2):
    md = gl.MeshData.sphere(rows=20, cols=20, radius=10)
    obj = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 0.1), shader='balloon', glOptions='additive')
    return obj

class MyWindow(QMainWindow):
    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
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
        label_euler_mode = QLabel()
        label_euler_mode.setText("euler angle type: ")
        self.rb_xyz = QRadioButton('xyz')
        self.rb_xyz.toggled.connect(self.rbcheck)
        self.rb_yzx = QRadioButton('yzx')
        self.rb_yzx.toggled.connect(self.rbcheck)
        self.rb_zxy = QRadioButton('zxy')
        self.rb_zxy.toggled.connect(self.rbcheck)
        self.rb_xzy = QRadioButton('xzy')
        self.rb_xzy.toggled.connect(self.rbcheck)
        self.rb_zyx = QRadioButton('zyx')
        self.rb_zyx.toggled.connect(self.rbcheck)
        self.rb_yxz = QRadioButton('yxz')
        self.rb_yxz.toggled.connect(self.rbcheck)

        self.rb_xyz.setChecked(True)
        self.rotate_mode = 'xyz'

        rb_layout = QHBoxLayout()
        rb_layout.addWidget(label_euler_mode)
        rb_layout.addWidget(self.rb_xyz)
        rb_layout.addWidget(self.rb_yzx)
        rb_layout.addWidget(self.rb_zxy)
        rb_layout.addWidget(self.rb_xzy)
        rb_layout.addWidget(self.rb_zyx)
        rb_layout.addWidget(self.rb_yxz)
        layout.addLayout(rb_layout)


        self.label_roll = QLabel()
        self.label_roll.setText("roll:  %3.2f dgree"%self.roll)
        self.label_pitch = QLabel()
        self.label_pitch.setText("pitch: %3.2f dgree"%self.pitch)
        self.label_yaw = QLabel()
        self.label_yaw.setText("yaw:   %3.2f dgree"%self.yaw)

        slider_roll = QSlider(QtCore.Qt.Horizontal)
        slider_roll.setMinimum(0)
        slider_roll.setMaximum(3600)
        slider_roll.valueChanged.connect(lambda val: self.setRotX(val))

        slider_pitch = QSlider(QtCore.Qt.Horizontal)
        slider_pitch.setMinimum(0)
        slider_pitch.setMaximum(3600)
        slider_pitch.valueChanged.connect(lambda val: self.setRotY(val))

        slider_yaw = QSlider(QtCore.Qt.Horizontal)
        slider_yaw.setMinimum(0)
        slider_yaw.setMaximum(3600)
        slider_yaw.valueChanged.connect(lambda val: self.setRotZ(val))
        layout.addWidget(self.label_roll)
        layout.addWidget(slider_roll)
        layout.addWidget(self.label_pitch)
        layout.addWidget(slider_pitch)
        layout.addWidget(self.label_yaw)
        layout.addWidget(slider_yaw)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.update)
    

        self.viewer.setWindowTitle('Euler rotation')
        self.viewer.setCameraPosition(distance=40)

        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(5, 5)
        self.viewer.addItem(g)

        ball2 = ball(8)
        org_arrow = GLArrowItem(size=10. , color=[0,0,1,1],  width=100)
        self.arrow = GLArrowItem(size=10. , color=[1,0,0,1], width=100)
        self.approx_arrow = GLArrowItem(size=10. , color=[1,0,1,1], width=100)
        self.viewer.addItem(org_arrow)
        self.viewer.addItem(ball2)
        self.viewer.addItem(self.arrow)
        self.viewer.addItem(self.approx_arrow)
        timer.start()

    def setRotX(self, val):
        self.roll = float(val/100.)
        self.label_roll.setText("roll:   %3.2f dgree"%self.roll)

    def setRotY(self, val):
        self.pitch = float(val/100.)
        self.label_pitch.setText("pitch: %3.2f dgree"%self.pitch)

    def setRotZ(self, val):
        self.yaw = float(val/100.)
        self.label_yaw.setText("yaw:   %3.2f dgree"%self.yaw)

    def rbcheck(self):
        if(self.rb_xyz.isChecked()):
            self.rotate_mode = 'xyz'
        elif(self.rb_yzx.isChecked()):
            self.rotate_mode = 'yzx'
        elif(self.rb_zxy.isChecked()):
            self.rotate_mode = 'zxy'
        elif(self.rb_xzy.isChecked()):
            self.rotate_mode = 'xzy'
        elif(self.rb_zyx.isChecked()):
            self.rotate_mode = 'zyx'
        elif(self.rb_yxz.isChecked()):
            self.rotate_mode = 'yxz'

    def update(self):
        roll = np.deg2rad(self.roll)
        pitch = np.deg2rad(self.pitch)
        yaw = np.deg2rad(self.yaw)
        R = euler2mat([roll, pitch, yaw], self.rotate_mode)
        T = np.eye(4)
        T[0:3, 0:3] = R
        self.arrow.setTransform(T)
        R = approx_euler2mat([roll, pitch, yaw], self.rotate_mode)
        T = np.eye(4)
        T[0:3, 0:3] = R
        self.approx_arrow.setTransform(T)
        self.viewer.update()

if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()
