import message_filters
from sensor_msgs.msg import Image
import cv2
import rospy
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import threading
from utilities.math_tools import *
from reprojection import *
import ros_numpy


class Frame:
    def __init__(self):
        self.pose = np.eye(4)
        self.feat_tracked = None
        self.feat_tracked_idx = None


def rgb2gray(im):
    im_gray_calc = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray_calc.astype(np.uint8)


def opticalFlowTrack(img0, img1, features0, back_check, horizontal_check):
    param = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    features1, status, error = cv2.calcOpticalFlowPyrLK(img0, img1, features0, (21, 21), 3)
    if back_check is True:
        reverse_features0, reverse_status, err = \
            cv2.calcOpticalFlowPyrLK(
                img1, img0, features1, (21, 21), 1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for i in range(status.shape[0]):
            if (status[i][0] and reverse_status[i][0] and np.linalg.norm(features0[i]-reverse_features0[i]) <= 0.5):
                status[i] = 1
            else:
                status[i] = 0
    if horizontal_check is True:
        for i in range(features0.shape[0]):
            if (status[i][0] == 0):
                continue
            diff = features0[i][0][1] - features1[i][0][1]
            horizontal_err = np.sqrt(diff*diff)
            dist = features0[i][0][0] - features1[i][0][0]
            if (dist < 0):
                status[i][0] = 0
                continue
            if (horizontal_err/dist > 0.1):
                status[i] = 0
    return features1, status


class TrackImage():
    def __init__(self):
        image_l_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        image_r_sub = message_filters.Subscriber('/kitti/camera_color_right/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_l_sub, image_r_sub], 10, 1/30.)
        self.img_old = None
        self.feat_tracked = None
        self.points = {}
        fx = 718.856
        fy = 718.856
        cx = 607.1928
        cy = 185.2157
        self.x_c1c2 = np.array([0, 0, 0, 0.2, 0, 0])
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        self.lock = threading.Lock()
        self.new_img = False
        self.frames = []
        ts.registerCallback(self.callback)

    def add_new_points(self, image_r, image_l):
        edge_mask = np.ones(image_r.shape, dtype=np.uint8) * 255

        if (self.feat_tracked is not None):
            for p in self.feat_tracked:
                cv2.circle(edge_mask, center=tuple(p[0].astype(int)), radius=20, color=0, thickness=-1)
    
        # find new feature.
        right_features = cv2.goodFeaturesToTrack(image_r, 200, 0.01, 20, mask=edge_mask)
        
        # ckeck feature in left image.
        left_features, status = opticalFlowTrack(image_r, image_l, right_features, False, True)
        right_features = right_features[np.where(status.flatten())]
        left_features = left_features[np.where(status.flatten())]

        # add new frame.
        frame = Frame()
        base_idx = 0
        if (self.feat_tracked is not None):
            last_frame = self.frames[-1]
            self.feat_tracked = np.vstack([self.feat_tracked, right_features])
            base_idx = last_frame.features_idx[-1] + 1
            frame.features = right_features
            frame.features_idx = np.append(last_frame.features_idx, 
                base_idx + np.arange(right_features.shape[0]))
        else:
            self.feat_tracked = right_features
            frame.features = right_features
            frame.features_idx = np.arange(right_features.shape[0])

        self.frames.append(frame)

        # add new world points.
        Kinv = np.linalg.inv(self.K)
        baseline = 0.2
        focal = self.K[0, 0]
        for i in range(right_features.shape[0]):
            u = right_features[i][0][0]
            v = right_features[i][0][1]
            disp = right_features[i][0][0] - left_features[i][0][0]
            p3d = Kinv.dot(np.array([u, v, 1.]))
            depth = (baseline * focal) / (disp)
            p3d *= depth
            self.points.update({i + base_idx: {'pos':p3d, 'frames:':[0], 'pt':[right_features[i][0]]}})


    def callback(self, image_r_msg, image_l_msg):
        if (self.new_img is True):
            return
        image_r = ros_numpy.numpify(image_r_msg)
        image_l = ros_numpy.numpify(image_l_msg)
        gray_r = rgb2gray(image_r)
        gray_l = rgb2gray(image_l)


        if self.img_old is None:
            self.add_new_points(gray_r, gray_l)
            self.img_old = [gray_r, gray_l]
            return

        img_cur = [gray_r, gray_l]
        self.feat_tracked_cur, status = opticalFlowTrack(self.img_old[0], img_cur[0], self.feat_tracked, True, False)
        # draw tracking
        self.draw(img_cur[0], status)

        self.feat_tracked = self.feat_tracked_cur[np.where(status.flatten())]

        frame = Frame()
        frame.features_idx = self.frames[-1].features_idx[np.where(status.flatten())]
        frame.features = self.feat_tracked
        self.frames.append(frame)

        self.add_new_points(gray_r, gray_l)

        self.img_old = img_cur


    def draw(self, img, status):
        image_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        print(self.feat_tracked.shape)
        for i in range(self.feat_tracked_cur.shape[0]):
            if (status[i][0] == 0):
                continue
            cv2.circle(image_show, tuple(self.feat_tracked_cur[i][0].astype(int)), 2, (255, 0, 0), 2)
            cv2.arrowedLine(image_show,
                            tuple(self.feat_tracked_cur[i][0].astype(int)),
                            tuple(self.feat_tracked[i][0].astype(int)), (0, 255, 0), 1, 8, 0, 0.2)
        cv2.imshow('image', image_show)
        cv2.waitKey(1)


if __name__ == '__main__':
    m = np.zeros([3, 3])
    m[0, 2] = 1.
    m[1, 0] = -1.
    m[2, 1] = -1.
    v = logSO3(m)
    args = rospy.myargv()
    rospy.init_node('controller_manager', anonymous=True)
    n = TrackImage()
    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        r.sleep()
