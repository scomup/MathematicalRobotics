import message_filters
from sensor_msgs.msg import Image
import cv2
import rospy
import ros_numpy
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import threading
from utilities.math_tools import *
from reprojection import *
from graph_optimization.graph_solver import *
from utilities.robust_kernel import *
from slam.demo_visual_slam import *
from visualization_msgs.msg import Marker
import quaternion

def opticalFlowTrack(img0,img1, pts0, back_check, horizontal_check):
    param = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    pts1, status, error = cv2.calcOpticalFlowPyrLK(img0, img1, pts0, (21, 21), 3)
    if back_check is True:
        reverse_pts0, reverse_status, err = cv2.calcOpticalFlowPyrLK(img1, img0, pts1, (21, 21), 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        for i in range(status.shape[0]): 
            if (status[i][0] and reverse_status[i][0] and np.linalg.norm(pts0[i]-reverse_pts0[i]) <= 0.5):
                status[i] = 1
            else:
                status[i] = 0
    if horizontal_check is True:
        for i in range(pts0.shape[0]): 
            if(status[i][0]==0):
                continue
            diff = pts0[i][0][1] - pts1[i][0][1]
            horizontal_err = np.sqrt(diff*diff)
            dist = pts0[i][0][0] - pts1[i][0][0]
            if (dist < 0):
                status[i][0] = 0
                continue
            if (horizontal_err/dist > 0.1):
                status[i] = 0
    return pts1, status



class testNode():
    def __init__(self):
        image_l_sub = message_filters.Subscriber('/oak_camera/left/image_rect', Image)
        image_r_sub = message_filters.Subscriber('/oak_camera/right/image_rect', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_l_sub, image_r_sub], 10, 1/30.)
        self.img0 = None
        self.img1 = None
        #self.pts = []
        self.pts0 = []
        self.points = None
        fx = 403.5362854003906
        fy = 403.4488830566406
        cx = 323.534423828125
        cy = 203.87405395507812
        self.x_c1c2 = np.array([0,0,0,0.075,0,0])
        self.x_bc = np.array([-1.20919958,  1.20919958, -1.20919958,0.0,0,0])
        self.K = np.array([[fx,0, cx],[0, fy,cy],[0,0,1.]])
        self.lock = threading.Lock()
        self.pub = rospy.Publisher("arrow_pub", Marker, queue_size = 10)
        self.new_img = False

        ts.registerCallback(self.callback)


    def pubMarker(self, x):
        R = expSO3(x[0:3])
        t = x[3:6]
        q = quaternion.from_rotation_matrix(R)
        marker_data = Marker()
        marker_data.header.frame_id = "map"
        marker_data.header.stamp = rospy.Time.now()
        marker_data.ns = "basic_shapes"
        marker_data.id = 0
        marker_data.action = Marker.ADD
        marker_data.pose.position.x = t[0]
        marker_data.pose.position.y = t[1]
        marker_data.pose.position.z = t[2]
        marker_data.pose.orientation.x=q.x
        marker_data.pose.orientation.y=q.y
        marker_data.pose.orientation.z=q.z
        marker_data.pose.orientation.w=q.w
        marker_data.color.r = 1.0
        marker_data.color.g = 0.0
        marker_data.color.b = 0.0
        marker_data.color.a = 1.0
        marker_data.scale.x = 1
        marker_data.scale.y = 1
        marker_data.scale.z = 1
        marker_data.lifetime = rospy.Duration()
        marker_data.type = 1
        self.pub.publish(marker_data)


    def initmap(self,image_r, image_l):
        new_pts = cv2.goodFeaturesToTrack(image_r, 200, 0.01, 20)
        right_pts, status = opticalFlowTrack(image_r, image_l,new_pts, False, True)
        self.pts0 = new_pts[np.where(status.flatten())]
        
        right_pts = right_pts[np.where(status.flatten())]

        Kinv = np.linalg.inv(self.K)
        baseline = 0.075
        focal = self.K[0,0]
        points = {}
        for i in range(self.pts0.shape[0]):
            u = self.pts0[i][0][0]
            v = self.pts0[i][0][1]
            disp = self.pts0[i][0][0] - right_pts[i][0][0]
            p3d_c = Kinv.dot(np.array([u,v,1.]))
            depth = (baseline * focal) / (disp)
            p3d_c *= depth
            p3d_b = transform(self.x_bc, p3d_c)
            p3d_w = p3d_b
            points.update({i:p3d_w})
        return points


    def calc_camera_pose(self):
        x_wc = np.zeros(6)
        self.pts1, status0 = opticalFlowTrack(self.img0[0], self.img1[0], self.pts0, True, False)
        right_pts, status1 = opticalFlowTrack(self.img1[0], self.img1[1], self.pts1, False, True)
        graph = graphSolver()
        idx = graph.addNode(camposeNode(x_wc)) 

        for i in range(self.pts1.shape[0]):
            if(status0[i][0] == 0 or status1[i][0] == 0):
                continue
            u0 = self.pts1[i][0]
            u1 = right_pts[i][0]
            p3d = self.points[i]
            idx_p = graph.addNode(featureNode(p3d),True) 
            graph.addEdge(reporjEdge(idx, idx_p, [self.x_c1c2, u0, u1, self.x_bc, self.K],kernel=CauchyKernel(0.5)))
        graph.solve(False, 0.1)
        return graph.nodes[idx].x


    def callback(self, image_r_msg, image_l_msg):
        if(self.new_img == True):
            return
        image_r = ros_numpy.numpify(image_r_msg)
        image_l = ros_numpy.numpify(image_l_msg)
        if self.img0 is None:
            self.points = self.initmap(image_r, image_l)
            self.img0 = [image_r, image_l]
        self.img1 = [image_r, image_l]
        self.new_img = True

    def run(self):
        if(self.new_img == False):
            return
        self.pts1, status = opticalFlowTrack(self.img0[0], self.img1[0], self.pts0, True, False)
        x = self.calc_camera_pose()
        print(x)
        self.pubMarker(x)
        self.draw(self.img1[0], status)
        self.new_img = False

    def draw(self, img, status):
        image_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for i in range(self.pts1.shape[0]):
            if(status[i][0] == 0):
                continue
            cv2.circle(image_show, tuple(self.pts1[i][0].astype(int)), 2, (255, 0, 0), 2)
            cv2.arrowedLine(image_show, tuple(self.pts1[i][0].astype(int)), tuple(self.pts0[i][0].astype(int)), (0, 255, 0), 1, 8, 0, 0.2)
        cv2.imshow('image', image_show)
        cv2.waitKey(1)
    


if __name__ == '__main__':
    m = np.zeros([3,3])
    m[0,2] = 1.
    m[1,0] = -1.
    m[2,1] = -1.
    v = logSO3(m)
    args = rospy.myargv()
    rospy.init_node('controller_manager', anonymous=True)
    n = testNode()
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        n.run()
        r.sleep()