
"""Kinova utility helpers replacing aloha_real/robot_utils.py."""
import time
from contextlib import contextmanager
from typing import Iterator
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge  # type: ignore

_bridge = CvBridge()

def ros_img_to_numpy(msg: Image) -> np.ndarray:
    """Convert ROS Image to BGR NumPy array."""
    return _bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

@contextmanager
def ros_rate(hz: float) -> Iterator[rospy.Rate]:
    r = rospy.Rate(hz)
    try:
        yield r
    finally:
        pass
