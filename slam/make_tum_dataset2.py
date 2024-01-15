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
from projection import *
from cv_bridge import CvBridge


fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField("r", 12, PointField.FLOAT32, 1),
          PointField("g", 16, PointField.FLOAT32, 1),
          PointField("b", 20, PointField.FLOAT32, 1)]


class Frame:
    def __init__(self):
        self.id = -1
        self.Twc = np.eye(4)
        self.keypoints = []
        self.descriptors = []
        self.img = None
        self.points_idx = []


class Point:
    def __init__(self):
        self.pw = np.eye(3)
        self.obs_by_frames = {}
        self.desc = None
        self.rgb = np.zeros(3)


def get_color(scalar, scalar_min=0, scalar_max=20):
    r = scalar_max - scalar_min
    v = (scalar - scalar_min) / r
    return (255 * (1 - v), 0, 255 * v)


def rgb2gray(im):
    im_gray_calc = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray_calc.astype(np.uint8)


def matching(query_frame, train_frame):
    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(np.array(query_frame.descriptors), np.array(train_frame.descriptors))
    return matches


class TrackingImage():
    def __init__(self):
        image_sub = message_filters.Subscriber('/camera/rgb/image_color', Image)
        depth_sub = message_filters.Subscriber('/camera/depth/image', Image)
        tf_sub = rospy.Subscriber('/tf', TFMessage, self.tfCB)
        self.pc_pub = rospy.Publisher("cloud", PointCloud2, queue_size=2)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=2)
        self.img_pub = rospy.Publisher('img', Image, queue_size=10)

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

    def add_new_frame(self, img, gray, depth):
        frame = Frame()
        frame.Twc = self.Twc
        frame.img = img
        frame.id = len(self.frames)

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        frame.points_idx = [-1] * len(keypoints)
        frame.keypoints = keypoints
        frame.descriptors = descriptors
        self.frames.append(frame)
        self.br = CvBridge()

    def callback(self, image_msg, depth_msg):

        if (self.Twc is None):
            return

        self.cnt += 1

        if(self.cnt % 10 != 0):
            return
        image = ros_numpy.numpify(image_msg)
        depth = ros_numpy.numpify(depth_msg)

        gray = rgb2gray(image)

        self.add_new_frame(image, gray, depth)

        # first frame
        if len(self.frames) <= 1:
            self.update_points([], depth, image)
            return
        last_frame = self.frames[-2]
        cur_frame = self.frames[-1]

        matches = matching(last_frame, cur_frame)
        # update points
        self.update_points(matches, depth, image)
        # draw tracking
        self.draw_tracking(last_frame, cur_frame)
        # self.draw_reproj()

        print(len(self.frames))
        if (len(self.frames) > self.max_frame_num):
            self.save("../data/ba/kitti_ba_dataset.txt")
            exit(0)

    def update_points(self, matches, depth, img):

        frame = self.frames[-1]

        if (len(self.frames) >= 2):
            last_frame = self.frames[-2]
            for i, m in enumerate(matches):
                pid = last_frame.points_idx[m.queryIdx]
                frame.points_idx[m.trainIdx] = pid
                kp = frame.keypoints[m.trainIdx]
                p = self.points[pid]
                # error = np.linalg.norm(project_error(frame.Twc, p.pw, np.array(kp.pt), self.K))
                # if (error > 10):
                #     frame.points_idx[m.trainIdx] = -1
                #     continue
                p.obs_by_frames.update({frame.id: kp})

        # update points
        Rwc, twc = makeRt(frame.Twc)
        # for i, kp, desc in zip(frame.points_idx, frame.keypoints, frame.descriptors):
        for i, kp in enumerate(frame.keypoints):
            idx = frame.points_idx[i]
            kp = frame.keypoints[i]
            desc = frame.descriptors[i]
            if (idx != -1):
                continue

            d = depth[int(kp.pt[1]), int(kp.pt[0])]

            if (np.isnan(d)):
                continue

            if (d < 0.3 and d > 5):
                continue

            point = Point()

            pc = self.Kinv @ np.array([kp.pt[0], kp.pt[1], 1.])
            pc = pc * d
            color = img[int(kp.pt[1]), int(kp.pt[0])].astype(np.float32)
            point.pw = pw = Rwc @ pc + twc
            # point.frames_kp.append(kp)
            point.rgb = color
            point.desc = desc
            point.obs_by_frames.update({frame.id: kp})
            frame.points_idx[i] = len(self.points)
            self.points.append(point)
        state = np.where(np.array(frame.points_idx) != -1)
        frame.points_idx = np.array(frame.points_idx)[state].tolist()
        frame.descriptors = frame.descriptors[state]
        frame.keypoints = np.array(frame.keypoints)[state].tolist()

    def draw_tracking(self, last_frame, cur_frame):
        frame = self.frames[-1]
        image_show = np.copy(frame.img)
        for kp, idx in zip(frame.keypoints, frame.points_idx):
            if(idx == -1):
                continue
            p = self.points[idx]
            try:
                kp_prv = p.obs_by_frames[cur_frame.id].pt
                kp_cur = p.obs_by_frames[last_frame.id].pt
                cv2.circle(image_show, (int(kp_cur[0]), int(kp_cur[1])), 2, (255, 0, 0), 2)
                cv2.arrowedLine(image_show,
                                (int(kp_prv[0]), int(kp_prv[1])),
                                (int(kp_cur[0]), int(kp_cur[1])), (0, 255, 0), 1, 8, 0, 0.2)
            except:
                pass
        self.img_pub.publish(self.br.cv2_to_imgmsg(image_show))

    def draw_reproj(self):
        frame = self.frames[-1]
        image_show = np.copy(frame.img)
        for kp, idx in zip(frame.keypoints, frame.points_idx):
            # idx = frame.points_idx[i]
            if(idx == -1):
                continue
            p = self.points[idx]
            Rcw, tcw = makeRt(np.linalg.inv(frame.Twc))
            pc = Rcw @ p.pw + tcw
            u_reproj = self.K @ pc
            u_reproj = u_reproj[:2] / u_reproj[2]
            u = kp.pt
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
            if (len(v.obs_by_frames) >= min_obs_num):
                points_lookup[i] = good_points
                good_points += 1
        obs = []
        for i, j in enumerate(points_lookup):
            if (j == -1):
                continue
            p = self.points[i]
            for k, kp in p.obs_by_frames.items():
                obs.append([k, j, kp.pt[0], kp.pt[1]])  # frame id, point id, u

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
    n.skip_frame = 10
    n.max_frame_num = 50
    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        r.sleep()
