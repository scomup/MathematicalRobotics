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


class Frame:
    def __init__(self):
        self.Twc = np.eye(4)
        self.us = []
        self.points_idx = []


class Point:
    def __init__(self):
        self.pw = np.eye(3)
        self.frames = []
        self.frames_u = []
        self.rgb = np.zeros(3)


def get_color(scalar, scalar_min=0, scalar_max=20):
    r = scalar_max - scalar_min
    v = (scalar - scalar_min) / r
    return (255 * (1 - v), 0, 255 * v)


def rgb2gray(im):
    im_gray_calc = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray_calc.astype(np.uint8)


def opticalFlowTrack(img0, img1, us0, back_check):
    us1, status, error = cv2.calcOpticalFlowPyrLK(img0, img1, np.array(us0).astype(np.float32), (21, 21), 3)
    if back_check is True:
        reverse_us0, reverse_status, err = \
            cv2.calcOpticalFlowPyrLK(
                img1, img0, np.array(us1).astype(np.float32), (21, 21), 1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for i in range(status.shape[0]):
            if (status[i][0] and reverse_status[i][0] and np.linalg.norm(us0[i]-reverse_us0[i]) <= 0.5):
                status[i] = 1
            else:
                status[i] = 0
    return us1, status


class TrackingImage():
    def __init__(self):
        image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image', Image)
        tf_sub = rospy.Subscriber('/tf', TFMessage, self.tfCB)
        self.pc_pub = rospy.Publisher("cloud", PointCloud2, queue_size=2)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=2)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1000, 1/30.)
        self.K = np.array([525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]).reshape([3, 3])
        self.Kinv = np.linalg.inv(self.K)
        self.Twc = None

        Rkc = quaternion_to_matrix(np.array([0.050, 0.868, -0.494, -0.029]))
        tkc = np.array([-0.000, 0.012, -0.094])
        self.Tkc = makeT(Rkc, tkc)

        self.gray_old = None
        self.points = []
        self.frames = []
        self.points_num = 0
        self.skip_point = 4
        self.min_obs_num = 5
        self.max_frame_num = 2
        self.cnt = 0
        self.skip_frame = 1

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

    def add_new_points(self, img, gray, depth, frame):
        mask_radius = 20

        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        if (len(frame.us) != 0):
            for u in frame.us:
                cv2.circle(mask, center=(int(u[0]), int(u[1])), radius=mask_radius, color=0, thickness=-1)

        edge = cv2.Canny(gray, 100, 200)
        depth_filter = np.logical_and(depth >= 0.3, depth < 5)
        edge_mask_filter = np.logical_and(edge == 255, mask == 255)

        tmp = np.where(np.logical_and(depth_filter, edge_mask_filter))

        us = np.vstack([tmp[1], tmp[0]]).T[::self.skip_point]
        depth_u = depth[us[:, 1], us[:, 0]]

        pc = (self.Kinv @ np.hstack([us, np.ones((len(us), 1))]).T).T
        pc = pc * depth_u[:, np.newaxis]

        colors = img[us[:, 1], us[:, 0]].astype(np.float32)
        colors = 255 - colors

        Rwc, twc = makeRt(self.Twc)
        pw = (Rwc @ pc.T).T + twc

        pw_with_rgb = np.hstack([pw, colors])

        header = Header()
        header.frame_id = "world"
        pmsg = point_cloud2.create_cloud(header, fields, pw_with_rgb)
        self.pc_pub.publish(pmsg)

        idx = 0
        for u, p, c in zip(us, pw, colors):
            point = Point()
            point.pw = p
            point.frames_u.append(u)
            point.rgb = c
            point.frames.append(len(self.frames))
            frame.us.append(u)
            frame.points_idx.append(len(self.points))
            self.points.append(point)

    def callback(self, image_msg, depth_msg):

        if (self.Twc is None):
            return

        self.cnt += 1

        if(self.cnt % 10 != 0):
            return
        image = ros_numpy.numpify(image_msg)
        depth = ros_numpy.numpify(depth_msg)
        gray = rgb2gray(image)

        if self.gray_old is None:
            frame = Frame()
            frame.Twc = self.Twc
            self.add_new_points(image, gray, depth, frame)
            self.gray_old = gray
            self.frames.append(frame)
            return
        last_frame = self.frames[-1]
        us_tracked_cur, status = opticalFlowTrack(self.gray_old, gray, last_frame.us, True)
        # draw tracking
        # self.draw_tracking(image, us_tracked_cur, last_frame.us, last_frame.points_idx, status)
        us_tracked = us_tracked_cur[np.where(status.flatten())]
        points_idx = np.array(last_frame.points_idx)[np.where(status.flatten())]

        # update points
        self.update_points(image, points_idx, us_tracked)
        # add new frame
        frame = Frame()
        frame.points_idx = points_idx.tolist()
        frame.us = us_tracked.tolist()
        # self.calc_camera_pose(frame, oxts)
        frame.Twc = self.Twc
        # self.draw_reproj(image, frame)

        # add new points to new frame
        self.add_new_points(image, gray, depth, frame)
        self.frames.append(frame)
        self.gray_old = gray
        print(len(self.frames))
        if (len(self.frames) > self.max_frame_num):
            self.save("../data/ba/kitti_ba_dataset.txt")
            exit(0)

    def update_points(self, image, feats_idx, feats_tracked):
        # update points
        for i, idx in enumerate(feats_idx):
            self.points[idx].frames.append(len(self.frames))
            self.points[idx].frames_u.append(feats_tracked[i])
            try:
                self.points[idx].rgb = image[tuple(feats_tracked[i][::-1].astype(int))].astype(float)
            except:
                pass

    def draw_tracking(self, img, feat_cur, feat_prv, feats_prv_idx, status):
        image_show = np.copy(img)
        for i in range(feat_cur.shape[0]):
            if (status[i][0] == 0):
                continue
            hint_frame_num = len(self.points[feats_prv_idx[i]].frames)
            color = get_color(hint_frame_num)
            cv2.circle(image_show, tuple(feat_cur[i].astype(int)), 2, color, 2)
            cv2.arrowedLine(image_show,
                            (int(feat_prv[i][0]), int(feat_prv[i][1])),
                            (int(feat_cur[i][0]), int(feat_cur[i][1])), (0, 255, 0), 1, 8, 0, 0.2)
        cv2.imshow('feature tracking', image_show)
        cv2.waitKey(1)

    def draw_reproj(self, img, frame):
        image_show = np.copy(img)
        for i, u in enumerate(frame.us):
            idx = frame.points_idx[i]
            p = self.points[idx]
            Rcw, tcw = makeRt(np.linalg.inv(frame.Twc))
            pc = Rcw @ p.pw + tcw
            u_reproj = self.K @ pc
            u_reproj = u_reproj[:2] / u_reproj[2]
            cv2.circle(image_show, (int(u[0]), int(u[1])), 2, (0, 255, 0), 2)
            cv2.circle(image_show, (int(u_reproj[0]), int(u_reproj[1])), 2, (255, 0, 0), 2)

            cv2.arrowedLine(image_show,
                            (int(u_reproj[0]), int(u_reproj[1])),
                            (int(u[0]), int(u[1])), (0, 255, 0), 1, 8, 0, 0.2)
        cv2.imshow('feature reprojection', image_show)
        cv2.waitKey(1)

    def save(self, fn):
        min_obs_num = np.min([self.min_obs_num, self.max_frame_num])
        points_lookup = np.full(len(self.points), -1)
        good_points = 0
        for i, v in enumerate(self.points):
            if (len(v.frames) >= min_obs_num):
                points_lookup[i] = good_points
                good_points += 1
        obs = []
        for i, j in enumerate(points_lookup):
            if (j == -1):
                continue
            p = self.points[i]
            for k, u in zip(p.frames, p.frames_u):
                obs.append([k, j, u[0], u[1]])  # frame id, point id, u

        cam_info = ' '.join(str(x) for x in [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], 0, 0])
        with open(fn, 'w') as file:
            header = [len(self.frames), good_points, len(obs)]
            header_str = ' '.join(str(x) for x in header) + '\n'
            file.writelines(header_str)

            for o in obs:
                o_str = ' '.join(str(x) for x in o) + '\n'
                file.writelines(o_str)

            for cam in self.frames:
                cam_str = ' '.join(str(x) for x in m2p(cam.Twc)) + ' ' + cam_info + '\n'
                file.writelines(cam_str)

            for i, j in enumerate(points_lookup):
                if (j == -1):
                    continue
                p = self.points[i]
                p_info = p.pw.tolist() + p.rgb.tolist()
                p_str = ' '.join(str(x) for x in p_info) + '\n'
                file.writelines(p_str)
            print(header)
            print("dataset is created!\n")


if __name__ == '__main__':
    rospy.init_node('TrackingImage', anonymous=True)
    n = TrackingImage()
    n.skip_frame = 20
    n.max_frame_num = 50
    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        r.sleep()
