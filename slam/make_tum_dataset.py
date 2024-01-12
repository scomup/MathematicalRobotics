import message_filters
from sensor_msgs.msg import Image
import cv2
import rospy
import ros_numpy
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import threading
# from utilities.math_tools import *
# from graph_optimization.graph_solver import *
# from utilities.robust_kernel import *
import rospy
from tf2_msgs.msg import TFMessage
# from tf.transformations import quaternion_matrix
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from utilities.math_tools import *


fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField("r", 12, PointField.FLOAT32, 1),
          PointField("g", 16, PointField.FLOAT32, 1),
          PointField("b", 20, PointField.FLOAT32, 1)]


def rainbow(scalars, scalar_min=0, scalar_max=5):
    range = scalar_max - scalar_min
    values = 1.0 - (scalars - scalar_min) / range
    # values = (scalars - scalar_min) / range  # using inverted color
    colors = np.zeros([scalars.shape[0], 3], dtype=np.float32)
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
    return colors * 255


def rgb2gray(im):
    im_gray_calc = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray_calc.astype(np.uint8)


class TrackingImage():
    def __init__(self):
        image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image', Image)
        tf_sub = rospy.Subscriber('/tf', TFMessage, self.tfCB)
        self.pc_pub = rospy.Publisher("cloud", PointCloud2, queue_size=2)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=2)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 1/30.)
        self.K = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]).reshape([3, 3])
        self.Kinv = np.linalg.inv(self.K)
        self.Twc = np.eye(4)

        Rkc = quaternion_to_matrix(np.array([0.050, 0.868, -0.494, -0.029]))
        tkc = np.array([-0.000, 0.012, -0.094])
        self.Tkc = makeT(Rkc, tkc)

        ts.registerCallback(self.callback)

    def tfCB(self, tf_msg):

        for tr in tf_msg.transforms:
            if (tr.header.frame_id == '/world' and tr.child_frame_id == '/kinect'):
                odom = Odometry()
                odom.header.frame_id = "world"
                q = tr.transform.rotation
                Rwk = quaternion_to_matrix(np.array([q.x, q.y, q.z, q.w]))
                twk = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
                Twk = makeT(Rwk, twk)
                self.Twc = Twk @ self.Tkc

                Rwc, twc = makeRt(self.Twc)
                qwc = matrix_to_quaternion(Rwc)
                odom.header.stamp = tr.header.stamp
                odom.pose.pose.position.x = twc[0]
                odom.pose.pose.position.y = twc[1]
                odom.pose.pose.position.z = twc[2]
                odom.pose.pose.orientation.x = qwc[0]
                odom.pose.pose.orientation.y = qwc[1]
                odom.pose.pose.orientation.z = qwc[2]
                odom.pose.pose.orientation.w = qwc[3]
                self.odom_pub.publish(odom)
                break

    def callback(self, image_msg, depth_msg):
        image = ros_numpy.numpify(image_msg)
        depth = ros_numpy.numpify(depth_msg)
        gray = rgb2gray(image)

        edge = cv2.Canny(gray, 100, 200)
        tmp = np.where(np.logical_and(np.logical_and(edge == 255, depth >= 0.3), depth < 5))

        depth_idx = np.vstack([tmp[1], tmp[0]]).T[::4]
        print(len(depth_idx))
        depth_filter = depth[depth_idx[:, 1], depth_idx[:, 0]]

        pc = (self.Kinv.dot(np.hstack([depth_idx, np.ones((len(depth_idx), 1))]).T)).T
        pc = pc * depth_filter[:, np.newaxis]

        rgb = image[depth_idx[:, 1], depth_idx[:, 0]].astype(np.float32)
        rgb = 255 - rgb

        Rwc, twc = makeRt(self.Twc)
        pw = (Rwc @ pc.T).T + twc

        pw_with_rgb = np.hstack([pw, rgb])

        header = Header()
        header.frame_id = "world"
        pmsg = point_cloud2.create_cloud(header, fields, pw_with_rgb)
        self.pc_pub.publish(pmsg)

        # color = rainbow(depth_filter)
        # image[depth_idx[:, 0], depth_idx[:, 1]] = color
        # cv2.imshow('feature tracking', image)
        # cv2.waitKey(1)


if __name__ == '__main__':
    rospy.init_node('TrackingImage', anonymous=True)
    n = TrackingImage()
    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        r.sleep()
