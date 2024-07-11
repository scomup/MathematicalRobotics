import pyqtgraph.opengl as gl
from OpenGL.GL import *
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def create_cube(size=2):
    vertexes = np.array([[1, 0, 0],  # 0
                         [0, 0, 0],  # 1
                         [0, 1, 0],  # 2
                         [0, 0, 1],  # 3
                         [1, 1, 0],  # 4
                         [1, 1, 1],  # 5
                         [0, 1, 1],  # 6
                         [1, 0, 1.]])  # 7
    vertexes *= size
    faces = np.array([[1, 0, 7], [7, 3, 1],
                      [1, 2, 4], [4, 0, 1],
                      [1, 2, 6], [6, 3, 1],
                      [0, 4, 5], [5, 7, 0],
                      [2, 4, 5], [5, 6, 2],
                      [3, 6, 5], [5, 7, 3]])
    colors = np.array([[1, 1, 0, 1], [1, 1, 0, 1],
                       [0, 1, 1, 1], [0, 1, 1, 1],
                       [1, 0, 1, 1], [1, 0, 1, 1],
                       [1, 0, 0, 1], [1, 0, 0, 1],
                       [0, 1, 0, 1], [0, 1, 0, 1],
                       [0, 0, 1, 1], [0, 0, 1, 1]])
    obj = gl.GLMeshItem(vertexes=vertexes, faces=faces, faceColors=colors, smooth=False)
    return obj


def create_ball(size=2):
    md = gl.MeshData.sphere(rows=100, cols=100, radius=10)
    obj = gl.GLMeshItem(meshdata=md, smooth=True, color=(0, 1, 0, 0.05), shader='balloon', glOptions='additive')
    return obj


class GLPlantItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, width=1, color=[1, 0, 0, 1],
                 pi=np.array([1, 0, 0.3, 0, 0.1, 0]), pj=np.array([0, 1, 0.3, -0.1, 0, 0])):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.width = width
        self.color = color
        self.points = []
        self.T = np.eye(4)
        size = 10.
        arr = []
        for a in np.arange(-size, size, 1):
            tmp = []
            for b in np.arange(-size, size, 1):
                x = pi * a + pj * b
                tmp.append(x[0:3])
            arr.append(tmp)
        self.points = np.array(arr)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        glLineWidth(self.width)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glBegin(GL_QUADS)
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])  # z is blue
        for i in range(self.points.shape[0]-1):
            for j in range(self.points.shape[1]-1):
                p00 = self.points[i, j]
                p01 = self.points[i, j+1]
                p10 = self.points[i+1, j]
                p11 = self.points[i+1, j+1]
                glVertex3f(p00[0], p00[1], p00[2])
                glVertex3f(p01[0], p01[1], p01[2])
                glVertex3f(p11[0], p11[1], p11[2])
                glVertex3f(p10[0], p10[1], p10[2])
        glEnd()
        glColor4f(self.color[0], self.color[1], self.color[2], 1)
        for i in range(self.points.shape[0]):
            glBegin(GL_LINE_STRIP)
            for j in range(self.points.shape[1]):
                p00 = self.points[i, j]
                glVertex3f(p00[0], p00[1], p00[2])
            glEnd()
        for i in range(self.points.shape[0]):
            glBegin(GL_LINE_STRIP)
            for j in range(self.points.shape[1]):
                p00 = self.points[j, i]
                glVertex3f(p00[0], p00[1], p00[2])
            glEnd()


class GLSurfItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, width=1, color=[1, 0, 0, 1],
                 pi=np.array([1, 0, 0.3, 0, 0.1, 0]), pj=np.array([0, 1, 0.3, -0.1, 0, 0])):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.width = width
        self.color = color
        self.points = []
        self.T = np.eye(4)
        size = 10.
        arr = []
        for a in np.arange(-size, size, 1):
            tmp = []
            for b in np.arange(-size, size, 1):
                x = pi * a + pj * b
                T = expSE3(x)
                tmp.append(T[0:3, 3])
            arr.append(tmp)
        self.points = np.array(arr)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        glLineWidth(self.width)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glBegin(GL_QUADS)
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])  # z is blue
        for i in range(self.points.shape[0]-1):
            for j in range(self.points.shape[1]-1):
                p00 = self.points[i, j]
                p01 = self.points[i, j+1]
                p10 = self.points[i+1, j]
                p11 = self.points[i+1, j+1]
                glVertex3f(p00[0], p00[1], p00[2])
                glVertex3f(p01[0], p01[1], p01[2])
                glVertex3f(p11[0], p11[1], p11[2])
                glVertex3f(p10[0], p10[1], p10[2])
        glEnd()
        glColor4f(self.color[0], self.color[1], self.color[2], 1)
        for i in range(self.points.shape[0]):
            glBegin(GL_LINE_STRIP)
            for j in range(self.points.shape[1]):
                p00 = self.points[i, j]
                glVertex3f(p00[0], p00[1], p00[2])
            glEnd()
        for i in range(self.points.shape[0]):
            glBegin(GL_LINE_STRIP)
            for j in range(self.points.shape[1]):
                p00 = self.points[j, i]
                glVertex3f(p00[0], p00[1], p00[2])
            glEnd()


class GLTrajItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, width=100, color=[1, 0, 0, 1], glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.width = width
        self.color = color
        self.points = []
        self.setGLOptions(glOptions)

    def addPoints(self, p):
        if (len(self.points) == 0):
            self.points.append(p)
            return
        l = self.points[-1]
        d = np.linalg.norm(l - p)
        if (d > 0.1):
            self.points.append(p)

    def clear(self):
        self.points.clear()

    def paint(self):
        glLineWidth(self.width)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glBegin(GL_LINE_STRIP)
        glColor4f(self.color[0], self.color[1], self.color[2], self.color[3])  # z is blue
        for p in self.points:
            glVertex3f(p[0], p[1], p[2])
        glEnd()


class GLAxisItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=[1, 1, 1], width=100, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        x, y, z = size
        self.width = width
        self.axis_x = np.array([x, 0, 0, 1])
        self.axis_y = np.array([0, y, 0, 1])
        self.axis_z = np.array([0, 0, z, 1])
        self.setGLOptions(glOptions)
        self.T = np.eye(4)

    def setTransform(self, T):
        self.T = T

    def paint(self):
        axis_x = self.T.dot(self.axis_x)
        axis_y = self.T.dot(self.axis_y)
        axis_z = self.T.dot(self.axis_z)
        self.setupGLState()
        glLineWidth(self.width)
        glBegin(GL_LINES)
        glColor4f(0, 0, 1, 1)  # z is blue
        glVertex3f(0, 0, 0)
        glVertex3f(axis_z[0], axis_z[1], axis_z[2])
        glColor4f(0, 1, 0, 1)  # y is green
        glVertex3f(0, 0, 0)
        glVertex3f(axis_y[0], axis_y[1], axis_y[2])
        glColor4f(1, 0, 0, 1)  # x is red
        glVertex3f(0, 0, 0)
        glVertex3f(axis_x[0], axis_x[1], axis_x[2])
        glEnd()


class GLRobotARMItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, Tba, colorA, colorB, glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.T = np.eye(4)
        self.A = np.eye(4)
        self.Tba = Tba
        self.Tab = np.linalg.inv(Tba)
        self.B = self.T @ self.Tab
        self.colorA = colorA
        self.colorB = colorB

    def setTransform(self, T):
        self.T = T

    def arm(self):
        glLineWidth(10)
        glBegin(GL_LINES)
        glColor4f(0.2, 0.2, 0.8, 0.1)  # z is blue
        glVertex4f(*self.A[:, 3])
        glVertex4f(*self.B[:, 3])
        glEnd()

    def getA(self):
        return (self.T)

    def getB(self):
        return (self.T @ self.Tab)

    def sphere(self, radius, slices, stacks, offset):
        """
        Draw a sphere with the given radius using triangles
        """
        for i in range(stacks):
            lat0 = np.pi * (-0.5 + float(i - 1) / stacks)
            z0 = radius * np.sin(lat0)
            zr0 = radius * np.cos(lat0)

            lat1 = np.pi * (-0.5 + float(i) / stacks)
            z1 = radius * np.sin(lat1)
            zr1 = radius * np.cos(lat1)

            glBegin(GL_TRIANGLE_STRIP)
            for j in range(slices):
                lng = 2 * np.pi * float(j - 1) / slices
                x = np.cos(lng)
                y = np.sin(lng)
                glVertex4f(x * zr0 + offset[0], y * zr0 + offset[1], z0 + offset[2], 1)
                glVertex4f(x * zr1 + offset[0], y * zr1 + offset[1], z1 + offset[2], 1)
            glEnd()

    def paint(self):
        self.setupGLState()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        # glEnable(GL_COLOR_MATERIAL)
        glMatrixMode(GL_MODELVIEW)
        glMultMatrixf(self.T.T)
        self.arm()
        glColor4f(*self.colorA)
        self.sphere(1, 50, 50, self.A[0:3, 3])
        glColor4f(*self.colorB)
        self.sphere(1, 50, 50, self.B[0:3, 3])
