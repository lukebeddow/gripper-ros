#!/usr/bin/env python3

import rospy
import sys
import numpy as np

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("panda_joint_node")
  rospy.loginfo("panda joint node has started")

  # create panda connection
  try:

    sys.path.insert(0, "/home/luke/franka_interface/build")
    import pyfranka_interface

    # create franka controller instance
    franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)

  except Exception as e:
    rospy.logerr(e)
    rospy.logerr("Failed to start panda conenction")

  rate = rospy.Rate(5)

  while not rospy.is_shutdown():

    print("Joint state is", franka_instance.getState().q)
    rate.sleep()
