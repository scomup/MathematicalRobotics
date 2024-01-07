from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator
import numpy as np
from PyQt5 import QtGui, QtCore


def rainbow(scalars, scalar_min=0, scalar_max=255, alpha=1):
    range = scalar_max - scalar_min
    values = 1.0 - (scalars - scalar_min) / range
    # values = (scalars - scalar_min) / range  # using inverted color
    colors = np.zeros([scalars.shape[0], 4], dtype=np.float32)
    values = np.clip(values, 0, 1)

    h = values * 5.0 + 1.0
    i = np.floor(h).astype(int)
    f = h - i
    f[np.logical_not(i % 2)] = 1 - f[np.logical_not(i % 2)]
    n = 1 - f

    # idx = i <= 1
    colors[i <= 1, 0] = n[i <= 1]
    colors[i <= 1, 1] = 0
    colors[i <= 1, 2] = 1

    colors[i == 2, 0] = 0
    colors[i == 2, 1] = n[i == 2]
    colors[i == 2, 2] = 1

    colors[i == 3, 0] = 0
    colors[i == 3, 1] = 1
    colors[i == 3, 2] = n[i == 3]

    colors[i == 4, 0] = n[i == 4]
    colors[i == 4, 1] = 1
    colors[i == 4, 2] = 0

    colors[i >= 5, 0] = 1
    colors[i >= 5, 1] = n[i >= 5]
    colors[i >= 5, 2] = 0
    colors[:, 3] = alpha
    return colors


class GL2DTextItem(gl.GLGraphicsItem.GLGraphicsItem):
    """Draws text over opengl 3D."""

    def __init__(self, **kwds):
        """All keyword arguments are passed to setData()"""
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.switch = 1
        self.settings = []
        self.settings.append({"name": "switch", "type": int, "set": self.setSwitch,  "get": self.getSwitch})

        self.pos = (100, 100)
        self.color = QtCore.Qt.GlobalColor.white
        self.text = ''
        self.font = QtGui.QFont('Helvetica', 16)
        self.setData(**kwds)

    def getSwitch(self):
        return self.switch

    def setSwitch(self, switch):
        self.switch = switch

    def setData(self, **kwds):
        args = ['pos', 'color', 'text', 'size', 'font']
        for k in kwds.keys():
            if k not in args:
                raise ValueError('Invalid keyword argument: %s (allowed arguments are %s)' % (k, str(args)))
        for arg in args:
            if arg in kwds:
                value = kwds[arg]
                if arg == 'pos':
                    self.pos = value
                elif arg == 'color':
                    value = value
                elif arg == 'font':
                    if isinstance(value, QtGui.QFont) is False:
                        raise TypeError('"font" must be QFont.')
                elif arg == 'size':
                    self.font.setPointSize(value)
                setattr(self, arg, value)
        self.update()
    
    def paint(self):
        if (self.switch == 0):
            return
        if len(self.text) < 1:
            return
        self.setupGLState()

        text_pos = QtCore.QPointF(*self.pos)
        painter = QtGui.QPainter(self.view())
        painter.setPen(self.color)
        painter.setFont(self.font)
        painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        painter.drawText(text_pos, self.text)
        painter.end()


class GLAxisItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, T=np.eye(4), size=1, width=100, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.width = width
        self.axis_x = np.array([size, 0, 0, 1])
        self.axis_y = np.array([0, size, 0, 1])
        self.axis_z = np.array([0, 0, size, 1])
        self.orig = np.array([0, 0, 0, 1])
        self.setGLOptions(glOptions)
        self.T = T

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
    def __init__(self, T=np.eye(4), size=1, width=1, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.width = width
        self.setGLOptions(glOptions)
        self.T = T

    def setTransform(self, T):
        self.T = T

    def paint(self):
        self.setupGLState()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLineWidth(self.width)
        glBegin(GL_LINES)
        hsize = self.size / 2

        # Draw the square base of the pyramid
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

class MyViewWidget(gl.GLViewWidget):
    def __init__(self):
        super(MyViewWidget, self).__init__()

    def mouseReleaseEvent(self, ev):
        if hasattr(self, 'mousePos'):
            delattr(self, 'mousePos')

    def mouseMoveEvent(self, ev):
        lpos = ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos
        if ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            pitch_abs = np.abs(self.opts['elevation'])
            camera_mode = 'view-upright'
            if(pitch_abs <= 45.0 or pitch_abs == 90):
                camera_mode = 'view'
            self.pan(diff.x(), diff.y(), 0, relative=camera_mode)

class BAViewer(QMainWindow):
    def __init__(self):
        super(BAViewer, self).__init__()
        self.setGeometry(0, 0, 1200, 800)
        self.initUI()
        self.cameras = {}

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)
        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = MyViewWidget()
        layout.addWidget(self.viewer, 1)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.update)

        self.viewer.setWindowTitle('Bundle Adjustment Viewer')
        self.viewer.setCameraPosition(distance=10)

        g = gl.GLGridItem()
        g.setSize(50, 50)
        g.setSpacing(1, 1)
        self.viewer.addItem(g)

        self.cloud = gl.GLScatterPlotItem(size=0.02, color=(1,1,1,0.5), pxMode=False)
        self.viewer.addItem(self.cloud)
        
        self.text = GL2DTextItem(text="", pos=(50, 50), size=20, color=QtCore.Qt.GlobalColor.white)
        self.viewer.addItem(self.text)

        axis = GLAxisItem(size=1, width=2)
        self.viewer.addItem(axis)

        timer.start()

    def update(self):
        self.viewer.update()

    def addItem(self, item):
        self.viewer.addItem(item)

    def clear(self):
        for item in self.items:
            try:
                self.viewer.removeItem(item)
            except:
                pass

    def setVertices(self, vertices):
        roll = np.pi/2
        R = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        T = np.eye(4)
        T[0:3,0:3] = R
        # T: transform camera coordinate to normal coordinate
        points = []
        for i, v in enumerate(vertices):
            if (type(v).__name__ == 'PointVertex'):
                points.append(v.x)
            elif (type(v).__name__ == 'CameraVertex'):
                pose = T @ np.linalg.inv(v.x)
                if not i in self.cameras:
                    cam_item = GLCameraFrameItem(T=pose, size=0.1, width=2)
                    self.addItem(cam_item)
                    self.cameras.update({i:cam_item})
                else:
                    self.cameras[i].setTransform(pose)
        points = np.array(points)
        points = (R @ points.T).T
        z = points[:,2]
        color = rainbow(z, scalar_min=-2, scalar_max=5, alpha=0.5)
        self.cloud.setData(pos=points, color=color)
        self.viewer.update()

    def setText(self, text):
        self.text.setData(text=text)

if __name__ == '__main__':
    app = QApplication([])
    cam = GLCameraFrameItem(size=1, width=1)
    viewer = BAViewer()
    viewer.addItem(cam)
    viewer.show()
    viewer.setText("test")
    app.exec_()
