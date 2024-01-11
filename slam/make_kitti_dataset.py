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
from os import listdir
from os.path import isfile, join
import struct
import time


def get_color(scalar, scalar_min=0, scalar_max=20):
    r = scalar_max - scalar_min
    v = (scalar - scalar_min) / r
    return (255 * (1 - v), 0, 255 * v)


def rainbow(scalars, scalar_min=0, scalar_max=255):
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


class Frame:
    def __init__(self):
        self.Tcw = np.eye(4)
        self.us = []
        self.points_idx = []


class Point:
    def __init__(self):
        self.pw = np.eye(3)
        self.frames = []
        self.frames_u = []
        self.rgb = np.zeros(3)


def read_bin(bin):
    points = np.fromfile(bin, dtype=np.float32).reshape([-1, 4])
    return points


def rgb2gray(im):
    im_gray_calc = 0.299 * im[:, :, 2] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 0]
    return im_gray_calc.astype(np.uint8)


def opticalFlowTrack(img0, img1, us0, back_check, horizontal_check):
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


class TrackImage:
    def __init__(self, path, s, num):
        self.img_folder = path + 'image_02/data/'
        self.vel_folder = path + 'velodyne_points/data/'
        self.oxts_folder = path + 'oxts/data/'
        self.img_files = sorted([f for f in listdir(self.img_folder) if isfile(join(self.img_folder, f))])[s:s+num]
        self.vel_files = sorted([f for f in listdir(self.vel_folder) if isfile(join(self.vel_folder, f))])[s:s+num]
        self.oxts_files = sorted([f for f in listdir(self.oxts_folder) if isfile(join(self.oxts_folder, f))])[s:s+num]

        self.gray_old = None
        # self.feat_tracked = None
        self.points = []
        self.K = np.eye(3)
        self.frames = []
        # self.kernel = HuberKernel(np.sqrt(5))
        self.kernel = HuberKernel(np.sqrt(3))
        self.points_num = 0
        self.skip = 1
        self.max_dist = 50
        self.Tcv = np.eye(4)
        self.Tci = np.eye(4)
        self.Twc = np.eye(4)

    def run(self):
        for ifn, vfn, ofn in zip(self.img_files, self.vel_files, self.oxts_files):
            img = cv2.imread(self.img_folder+ifn)
            vel = read_bin(self.vel_folder+vfn).astype(np.float32)
            vel[:, 3] = 1
            oxts = np.loadtxt(self.oxts_folder+ofn)
            cloud = (self.Tcv @ vel.T).T[:, :3]
            self.run_once(img, cloud, oxts)

    def add_new_points(self, img, cloud, frame):
        mask_radius = 20

        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        if (len(frame.us) != 0):
            for u in frame.us:
                cv2.circle(mask, center=(int(u[0]), int(u[1])), radius=mask_radius, color=0, thickness=-1)

        h = img.shape[0]
        w = img.shape[1]

        image_show = np.copy(img)
        board = 40

        cloud = cloud[::self.skip]
        colors = rainbow(cloud[:, 2], 0, self.max_dist)

        idx = 0
        for pc, color in zip(cloud, colors):
            u = project(pc, self.K)
            idx += 1

            dist = np.linalg.norm(pc)
            if (u[0] <= board or u[0] > w - board):
                continue
            if (u[1] <= board or u[1] > h - board):
                continue
            if mask[int(u[1]), int(u[0])] == 0:
                continue
            # only in front of the camera
            if(pc[2] < 0):
                continue
            if(dist > self.max_dist):
                continue

            point = Point()
            Rwc, twc = makeRt(np.linalg.inv(frame.Tcw))
            point.pw = Rwc @ pc + twc
            point.frames.append(len(self.frames))
            point.frames_u.append(u)
            point.rgb = img[int(u[1]), int(u[0])]

            frame.us.append(u)
            frame.points_idx.append(len(self.points))

            self.points.append(point)

            # cv2.circle(image_show, (int(u[0]), int(u[1])), 2, (int(color[0]), int(color[1]), int(color[2])), 2)
            # image_show[int(u[1]), int(u[0])] = color.astype(np.uint8)

        # cv2.imshow('draw_new', image_show)
        # cv2.waitKey(0)

    def update_points(self, image, feats_idx, feats_tracked):
        # update points
        for i, idx in enumerate(feats_idx):
            self.points[idx].frames.append(len(self.frames))
            self.points[idx].frames_u.append(feats_tracked[i])
            try:
                self.points[idx].rgb = image[tuple(feats_tracked[i][::-1].astype(int))].astype(float)
            except:
                pass

    def calc_camera_pose(self, frame, oxts):
        """
        Tcw = np.linalg.inv(self.Twc)
        graph = GraphSolver()
        # -----------------
        # Calcuate the pose of new frame
        # -----------------
        # Add camera vertex
        graph.add_vertex(CameraVertex(Tcw))

        all_frames = set()
        # Add point vertex
        for i, idx in enumerate(frame.points_idx):
            pw = self.points[idx].pw
            all_frames.update(self.points[idx].frames)
            pidx = graph.add_vertex(PointVertex(pw), is_constant=True)
            u = frame.us[i]
            r = project_error0(Tcw, pw, u, self.K)
            graph.add_edge(ProjectEdge([0, pidx], [u, self.K], np.eye(2), self.kernel))
        # solve
        graph.solve(True)

        frame.Tcw = graph.vertices[0].x
        self.Twc = np.linalg.inv(frame.Tcw)
        """
        dt = 0.1
        Rwc, twc = makeRt(self.Twc)
        Rci = self.Tci[0:3, 0:3]
        vi = np.array(oxts[8:11])
        omgi = np.array(oxts[17:20])
        vc = Rci @ vi
        omgc = Rci @ omgi

        twc = twc + (Rwc @ vc) * dt
        Rwc = Rwc @ expSO3(omgc * dt)

        self.Twc = makeT(Rwc, twc)
        frame.Tcw = np.linalg.inv(self.Twc)

    def run_once(self, image, cloud, oxts):
        gray = rgb2gray(image)
        # first frame.
        if self.gray_old is None:
            frame = Frame()
            self.add_new_points(image, cloud, frame)
            self.gray_old = gray
            self.frames.append(frame)
            return
        last_frame = self.frames[-1]
        us_tracked_cur, status = opticalFlowTrack(self.gray_old, gray, last_frame.us, True, False)

        # draw tracking
        self.draw_tracking(image, us_tracked_cur, last_frame.us, last_frame.points_idx, status)

        us_tracked = us_tracked_cur[np.where(status.flatten())]
        points_idx = np.array(last_frame.points_idx)[np.where(status.flatten())]

        # update points
        self.update_points(image, points_idx, us_tracked)

        # add new frame
        frame = Frame()
        frame.points_idx = points_idx.tolist()
        frame.us = us_tracked.tolist()
        self.calc_camera_pose(frame, oxts)
        self.draw_reproj(image, frame)

        # add new points to new frame
        self.add_new_points(image, cloud, frame)
        self.frames.append(frame)
        self.gray_old = gray

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
            Rcw, tcw = makeRt(frame.Tcw)
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

    def save(self, fn, min_obs_num):
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
        # len(self.frames), good_points, len(obs)

        cam_info = ' '.join(str(x) for x in [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], 0, 0])
        with open(fn, 'w') as file:
            header = [len(self.frames), good_points, len(obs)]
            header_str = ' '.join(str(x) for x in header) + '\n'
            file.writelines(header_str)

            for o in obs:
                o_str = ' '.join(str(x) for x in o) + '\n'
                file.writelines(o_str)

            for cam in self.frames:
                cam_str = ' '.join(str(x) for x in m2p(cam.Tcw)) + ' ' + cam_info + '\n'
                file.writelines(cam_str)

            for i, j in enumerate(points_lookup):
                if (j == -1):
                    continue
                p = self.points[i]
                p_info = p.pw.tolist() + p.rgb.tolist()
                p_str = ' '.join(str(x) for x in p_info) + '\n'
                file.writelines(p_str)

if __name__ == '__main__':
    args = rospy.myargv()

    path = '/home/liu/bag/kitti/2011_10_03/2011_10_03_drive_0027_sync/'
    start_frame = 130
    frame_num = 100
    # set the camera info
    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy = 185.2157

    n = TrackImage(path=path, s=start_frame, num=frame_num)
    n.skip = 16
    # set the vel to cam matrix
    # get the data from kitti calib_velo_to_cam.txt
    Rcv = np.array([7.967514e-03, -9.999679e-01, -8.462264e-04,
                    -2.771053e-03, 8.241710e-04, -9.999958e-01,
                    9.999644e-01, 7.969825e-03, -2.764397e-03]).reshape([3, 3])
    tcv = np.array([-1.377769e-02, -5.542117e-02, -2.918589e-01])
    Tcv = makeT(Rcv, tcv)
    n.Tcv = Tcv
    # set the vel to cam matrix
    # get the data from kitti calib_imu_to_velo.txt
    Rvi = np.array([9.999976e-01, 7.553071e-04, -2.035826e-03,
                    -7.854027e-04, 9.998898e-01, -1.482298e-02,
                    2.024406e-03, 1.482454e-02, 9.998881e-01]).reshape([3, 3])
    tvi = np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
    Tvi = makeT(Rvi, tvi)
    Tci = Tcv @ Tvi
    n.Tci = Tci

    n.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])

    n.run()
    n.save('data/ba/kitti_ba_dataset.txt', 5)
    print(len(n.points))
