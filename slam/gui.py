from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel, QRadioButton, QApplication
from OpenGL.GL import *
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QKeyEvent, QIntValidator, QDoubleValidator
import numpy as np
from PyQt5 import QtGui, QtCore
import threading
import time

CAPACITY = 10000000




class CloudPlotItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, **kwds):
        super().__init__()
        glopts = kwds.pop('glOptions', 'additive')
        self.setGLOptions(glopts)
        self.points_capacity = 0
        self.valid_point_num = 0
        self.valid_point_num_in_buf = 0  # the number of point have been upload to gpu buffer
        self.pos = np.empty((0, 3), np.float32)
        self.color = np.empty((0, 4), np.float32)
        self.use_color_buffer = False
        self.flat_color = [1, 1, 1]
        self.alpha = 1.
        self.size = 10
        self.mutex = threading.Lock()
        self.need_update_buffer_capacity = True
        self.need_update_buffer = True
        self.setData(**kwds)

    def setSize(self, size):
        self.size = size

    def getSize(self):
        return self.size

    def clear(self):
        self.mutex.acquire()
        self.valid_point_num = 0
        self.valid_point_num_in_buf = 0
        self.need_update_buffer = True
        self.mutex.release()

    def setAlpha(self, alpha):
        self.alpha = alpha
        if self.use_color_buffer:
            self.color[:, 3] = self.alpha
            glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
            glBufferData(GL_ARRAY_BUFFER, self.color.nbytes, self.color, GL_DYNAMIC_DRAW)

    def getAlpha(self):
        return self.alpha

    def setData(self, **kwds):
        self.mutex.acquire()
        if 'pos' in kwds:
            pos = kwds.pop('pos')
            self.pos = np.ascontiguousarray(pos, dtype=np.float32)
            self.valid_point_num = pos.shape[0]
            self.need_update_buffer_capacity = True
        if 'alpha' in kwds:
            self.alpha = kwds.pop('alpha')
        if 'flat_color' in kwds:
            self.flat_color = kwds.pop('flat_color')
        if 'color' in kwds:
            self.color = kwds.pop('color')
            self.color[:, 3] = self.alpha
            self.use_color_buffer = True
        if 'size' in kwds:
            self.size = kwds.pop('size')
        self.need_update_buffer = True
        self.mutex.release()

    def appendData(self, pos, color, update_buffer=True):
        self.mutex.acquire()
        # time_start = time.time()
        p_size = pos.shape[0]
        if (self.valid_point_num + p_size > self.points_capacity):
            # update cpu buffer capacity size
            self.points_capacity += CAPACITY
            print("update cpu buffer capacity to %d points." % self.points_capacity)
            new_pos = np.empty((self.points_capacity, 3), np.float32)
            new_color = np.empty((self.points_capacity, 4), np.float32)
            new_pos[0:self.valid_point_num, :] = self.pos[0:self.valid_point_num, :]
            new_color[0:self.valid_point_num, :] = self.color[0:self.valid_point_num, :]
            self.pos = new_pos
            self.color = new_color
            self.need_update_buffer_capacity = True  # gpu buffer capacity
        self.pos[self.valid_point_num:self.valid_point_num + p_size] = pos
        if(color.shape[0] == p_size):
            color[:, 3] = self.alpha
            self.color[self.valid_point_num:self.valid_point_num + p_size] = color
            self.use_color_buffer = True
        else:
            self.use_color_buffer = False
        self.valid_point_num += p_size
        self.need_update_buffer = update_buffer
        # time_end = time.time()
        # elapsed = time_end - time_start
        # print("appendData %f [ms]" % (elapsed * 1000.))
        self.mutex.release()

    def updateRenderBuffer(self):
        if(not self.need_update_buffer):
            return
        self.mutex.acquire()
        # Create a vertex buffer object
        if self.need_update_buffer_capacity:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.pos.nbytes, self.pos, GL_DYNAMIC_DRAW)
            # Create a color buffer object
            if self.use_color_buffer:
                glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
                glBufferData(GL_ARRAY_BUFFER, self.color.nbytes, self.color, GL_DYNAMIC_DRAW)
            self.need_update_buffer_capacity = False
        else:
            new_point_num = self.valid_point_num - self.valid_point_num_in_buf
            # Add new pos to object buffer
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferSubData(GL_ARRAY_BUFFER, self.valid_point_num_in_buf * 3 * 4,
                            new_point_num * 3 * 4, self.pos[self.valid_point_num_in_buf:, :])
            if self.use_color_buffer:
                # Add new color to color buffer
                glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
                glBufferSubData(GL_ARRAY_BUFFER, self.valid_point_num_in_buf * 4 * 4,
                                new_point_num * 4 * 4, self.color[self.valid_point_num_in_buf:, :])
        self.valid_point_num_in_buf = self.valid_point_num
        self.need_update_buffer = False
        self.mutex.release()

    def initializeGL(self):
        self.vbo = glGenBuffers(1)
        self.cbo = glGenBuffers(1)

    def paint(self):
        self.setupGLState()
        if self.valid_point_num == 0:
            return
        self.updateRenderBuffer()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if(self.use_color_buffer):
            glBindBuffer(GL_ARRAY_BUFFER, self.cbo)
            glColorPointer(4, GL_FLOAT, 0, None)
            glEnableClientState(GL_COLOR_ARRAY)
        else:
            glColor4f(self.flat_color[0], self.flat_color[1], self.flat_color[2], self.alpha)

        # draw points
        glPointSize(self.size)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_POINTS, 0, self.valid_point_num_in_buf)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        if(self.use_color_buffer):
            glDisableClientState(GL_COLOR_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, 0)


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
        frame_points = np.array([[-hsize, -hsize, 0, 1],
                                [hsize, -hsize, 0, 1],
                                [hsize, hsize, 0, 1],
                                [-hsize, hsize, 0, 1],
                                [0, 0, hsize, 1]])
        frame_points = (self.T @ frame_points.T).T[:, 0:3]
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
        self.viewer.setWindowTitle('Bundle Adjustment Viewer')
        self.viewer.setBackgroundColor(255, 255, 255, 255)
        self.viewer.setCameraParams(distance=5, center=QtGui.QVector3D(0, 0, 0), azimuth=-43, elevation=20)

        layout.addWidget(self.viewer, 1)

        g = gl.GLGridItem()
        g.setSize(50, 50)
        g.setSpacing(1, 1)
        self.viewer.addItem(g)

        self.cloud = CloudPlotItem(size=3, alpha=0.2, flat_color=[0, 0, 0])
        self.viewer.addItem(self.cloud)

        self.text = GL2DTextItem(text="", pos=(50, 50), size=20, color=QtCore.Qt.GlobalColor.black)
        self.viewer.addItem(self.text)

        axis = GLAxisItem(size=1, width=2)
        self.viewer.addItem(axis)

    def addItem(self, item):
        self.viewer.addItem(item)

    def clear(self):
        for item in self.items:
            try:
                self.viewer.removeItem(item)
            except:
                pass

    def setVertices(self, vertices):
        T = np.eye(4)
        roll = np.pi/2
        R = np.array([[1, 0, 0],
                     [0, np.cos(roll), -np.sin(roll)],
                     [0, np.sin(roll), np.cos(roll)]])
        T[0:3, 0:3] = R
        points = []
        for i, v in enumerate(vertices):
            if (type(v).__name__ == 'PointVertex'):
                points.append(v.x)
            elif (type(v).__name__ == 'CameraVertex'):
                pose = T @ v.x
                if i not in self.cameras:
                    cam_item = GLCameraFrameItem(T=pose, size=0.05, width=2)
                    self.addItem(cam_item)
                    self.cameras.update({i: cam_item})
                else:
                    self.cameras[i].setTransform(pose)
        points = np.array(points)
        points = (T[0:3, 0:3] @ points.T).T
        z = points[:, 2]

        self.cloud.setData(pos=points.astype(np.float32))
        # self.viewer.update()

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
