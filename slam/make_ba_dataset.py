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
from projection import *
import ros_numpy
from gui import *
from threading import Thread


class Frame:
    def __init__(self):
        self.Twc = np.eye(4)
        self.feats = None
        self.feats_idx = None

class Point:
    def __init__(self):
        self.pw = np.eye(3)
        self.frames = []
        self.rgb = np.zeros(3)


def get_color(scalar, scalar_min=0, scalar_max=20):
    r = scalar_max - scalar_min
    v = (scalar - scalar_min) / r
    return (255 * (1 - v), 0, 255 * v)


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
            if (dist < 10):
                status[i][0] = 0
                continue
            if (horizontal_err/dist > 0.1):
                status[i] = 0
    return features1, status


class TrackImage:
    def __init__(self, viewer):
        image_l_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image)
        image_r_sub = message_filters.Subscriber('/kitti/camera_color_right/image_raw', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_l_sub, image_r_sub], 10, 1/30.)
        self.imgs_old = None
        # self.feat_tracked = None
        self.points = {}
        fx = 718.856
        fy = 718.856
        cx = 607.1928
        cy = 185.2157
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        self.frames = []
        self.kernel = HuberKernel(np.sqrt(5))
        self.viewer = viewer
        self.cam = GLCameraFrameItem(size=0.2, width=1)
        self.viewer.addItem(self.cam)

        ts.registerCallback(self.callback)

    def add_new_points(self, imgs, frame):
        new_points_num = 400
        radius = 20

        # if (frame.feats is not None and frame.feats.shape[0] > new_points_num/2):
        #     return

        image_r, image_l = imgs
        edge_mask = np.ones(image_r.shape, dtype=np.uint8) * 255

        if (frame.feats is not None):
            new_points_num -= frame.feats.shape[0]
            for p in frame.feats:
                cv2.circle(edge_mask, center=tuple(p[0].astype(int)), radius=radius, color=0, thickness=-1)

        if (new_points_num <= 0):
            return
        # find new feature.
        feats_r = cv2.goodFeaturesToTrack(image_r, new_points_num, 0.01, minDistance=radius, mask=edge_mask)

        # ckeck feature in left image.
        feats_l, status = opticalFlowTrack(image_r, image_l, feats_r, False, True)
        feats_r = feats_r[np.where(status.flatten())]
        feats_l = feats_l[np.where(status.flatten())]

        # add new feats to frame.
        base_idx = 0
        if (frame.feats is not None):
            print("add %d new points\n"%feats_r.shape[0])
            frame.feats = np.vstack([frame.feats, feats_r])
            base_idx = frame.feats_idx[-1] + 1
            frame.feats_idx = np.append(frame.feats_idx, base_idx + np.arange(feats_r.shape[0]))
        else:
            frame.feats = feats_r
            frame.feats_idx = np.arange(feats_r.shape[0])

        # add new world points.
        Kinv = np.linalg.inv(self.K)
        baseline = 0.2
        focal = self.K[0, 0]
        for i in range(feats_r.shape[0]):
            u = feats_r[i][0][0]
            v = feats_r[i][0][1]
            disp = feats_r[i][0][0] - feats_l[i][0][0]
            pc = Kinv.dot(np.array([u, v, 1.]))
            depth = (baseline * focal) / (disp)
            pc *= depth
            point = Point()
            point.frames.append(len(self.frames))
            Rwc, twc = makeRt(frame.Twc)
            point.pw = Rwc @ pc + twc
            point.rgb = self.imgs_color[0][tuple(feats_r[i][0][::-1].astype(np.int32))].astype(float)
            self.points.update({i + base_idx: point})

    def update_points(self, feats_idx, feats_tracked):
        # update points
        for i, idx in enumerate(feats_idx):
            self.points[idx].frames.append(len(self.frames))
            try:
                self.points[idx].rgb = self.imgs_color[0][tuple(feats_tracked[i][0][::-1].astype(np.int32))].astype(float)
            except:
                pass

    def calc_camera_pose(self, guess_Twc, frame):
        Tcw = np.linalg.inv(guess_Twc)
        graph = GraphSolver()

        # Add camera vertex
        graph.add_vertex(CameraVertex(Tcw))
    
        #Add point vertex
        for i, idx in enumerate(frame.feats_idx):
            pw = self.points[idx].pw
            pidx = graph.add_vertex(PointVertex(pw), is_constant=True)
            u = frame.feats[i][0]
            graph.add_edge(ProjectEdge([0, pidx], [u, self.K], np.eye(2), None)) 

        # solve
        graph.solve(True, 0.1)
        new_Tcw = graph.vertices[0].x
        frame.Twc = np.linalg.inv(new_Tcw)
        roll = -np.pi/2
        R = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        T = np.eye(4)
        T[0:3,0:3] = R

        self.cam.setTransform(T @ frame.Twc)

    def callback(self, image_r_msg, image_l_msg):
        print("here!")
        image_r = ros_numpy.numpify(image_r_msg)
        image_l = ros_numpy.numpify(image_l_msg)
        gray_r = rgb2gray(image_r)
        gray_l = rgb2gray(image_l)
        self.imgs_color = [image_r, image_l]
        imgs = [gray_r, gray_l]
        # first frame.
        if self.imgs_old is None:
            frame = Frame()
            self.add_new_points(imgs, frame)
            self.imgs_old = imgs
            self.frames.append(frame)
            return
        last_frame = self.frames[-1]
        feat_tracked_cur, status = opticalFlowTrack(self.imgs_old[0], imgs[0], last_frame.feats, True, False)

        # draw tracking
        # self.draw_tracking(imgs[0], feat_tracked_cur, last_frame.feats, last_frame.feats_idx, status)

        feat_tracked = feat_tracked_cur[np.where(status.flatten())]
        feats_idx = last_frame.feats_idx[np.where(status.flatten())]

        # update points
        self.update_points(feats_idx, feat_tracked)

        # add new frame
        frame = Frame()
        frame.feats_idx = feats_idx
        frame.feats = feat_tracked
        self.calc_camera_pose(last_frame.Twc, frame)

        # add new points to new frame
        self.add_new_points(imgs, frame)
        self.frames.append(frame)
        self.imgs_old = imgs
        
        points = []
        for p in self.points:
            points.append(self.points[p].pw)
        points = np.array(points)
        roll = -np.pi/2
        R = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
        self.viewer.cloud.setData(pos=(R @ points.T).T)


    def draw_tracking(self, img, feat_cur, feat_prv, feats_prv_idx, status):
        image_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image_show = np.copy(self.imgs_color[0])
        for i in range(feat_cur.shape[0]):
            if (status[i][0] == 0):
                continue
            hint_frame_num = len(self.points[feats_prv_idx[i]].frames)
            color = get_color(hint_frame_num)
            cv2.circle(image_show, tuple(feat_cur[i][0].astype(int)), 2, color, 2)
            cv2.arrowedLine(image_show,
                            tuple(feat_cur[i][0].astype(int)),
                            tuple(feat_prv[i][0].astype(int)), (0, 255, 0), 1, 8, 0, 0.2)
        cv2.imshow('image', image_show)
        cv2.waitKey(1)


if __name__ == '__main__':
    args = rospy.myargv()
    rospy.init_node('controller_manager', anonymous=True)
    app = QApplication([])
    viewer = BAViewer()
    viewer.show()
    n = TrackImage(viewer)
    app.exec_()

