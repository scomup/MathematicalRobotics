from gui import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.math_tools import *


def approx_euler2mat(euler, order='xyz'):
    euler = np.array(euler)
    return skew(euler) + np.eye(3)


class XPlane(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=10., x=10., glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.size = size
        self.x = x
        self.setGLOptions(glOptions)

    def paint(self):
        s = self.size
        x = self.x
        self.setupGLState()
        glBegin(GL_QUADS)
        glColor4f(0, 1, 0, 0.2)  # z is blue
        glVertex3d(x, s, -s)
        glVertex3d(x, s, s)
        glVertex3d(x, -s, s)
        glVertex3d(x, -s, -s)
        glEnd()


class Gui3d_sr(Gui3d):
    def __init__(self, static_obj, dynamic_obj):
        super().__init__(static_obj, dynamic_obj)

    def update(self):
        roll = np.deg2rad(self.roll)
        pitch = np.deg2rad(self.pitch)
        yaw = np.deg2rad(self.yaw)
        R = euler2mat([roll, pitch, yaw], self.rotate_mode)
        T = np.eye(4)
        T[0:3, 0:3] = R
        self.dynamic_obj[0].setTransform(T)
        R = approx_euler2mat([roll, pitch, yaw], self.rotate_mode)
        T = np.eye(4)
        T[0:3, 0:3] = R
        self.dynamic_obj[1].setTransform(T)
        self.viewer.update()

if __name__ == '__main__':
    app = QApplication([])
    ball = create_ball(8)
    org_arrow = GLArrowItem(size=10., color=[0, 0, 1, 1],  width=5)
    arrow = GLArrowItem(size=10., color=[1, 0, 0, 1], width=5)
    approx_arrow = GLArrowItem(size=10., color=[1, 0, 1, 1], width=5)

    n_x, n_y = 128, 128
    xy_init = 1.0, 4.0
    x_range = -10, 10
    y_range = -10, 10
    x = np.linspace(x_range[0], x_range[1], n_x)
    y = np.linspace(y_range[0], y_range[1], n_y)

    rad = 0.5
    x_grid, y_grid = np.meshgrid(y, x)
    z = 3 * np.exp(-((x_grid-xy_init[0])**2.0)*rad**2) \
          * np.exp(-((y_grid-xy_init[1])**2.0)*rad**2)

    p = XPlane()
    window = Gui3d_sr(static_obj=[org_arrow, ball, p], dynamic_obj=[arrow, approx_arrow])
    window.show()
    app.exec_()
