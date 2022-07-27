import message_filters
from sensor_msgs.msg import Image
#import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
#from cv_bridge import CvBridge
import rospy
import ros_numpy
import matplotlib.pyplot as plt


class testNode():
    def __init__(self):
        image_l_sub = message_filters.Subscriber('/oak_camera/left/image_rect', Image)
        image_r_sub = message_filters.Subscriber('/oak_camera/right/image_rect', Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_l_sub, image_r_sub], 10, 1/30.)
        ts.registerCallback(self.callback)
        self.img0 = None
        self.img = None

    def callback(self, image_r_msg, image_l_msg):
        image_r = ros_numpy.numpify(image_r_msg)
        image_l = ros_numpy.numpify(image_l_msg)
        if self.img0 is None:
            self.img0 = [image_r, image_l]
        self.img = [image_r, image_l]

    def run(self):
        if self.img is None:
            return
        cv2.imshow('image', self.img[0])
        cv2.waitKey(1)




if __name__ == '__main__':
    args = rospy.myargv()
    rospy.init_node('controller_manager', anonymous=True)
    n = testNode()
    while not rospy.is_shutdown():
        n.run()