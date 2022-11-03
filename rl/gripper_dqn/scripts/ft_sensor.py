#!/usr/bin/env python3

import rospy
# from gripper_msgs.msg import FloatXYZ
from subprocess import Popen, PIPE, STDOUT

if __name__ == "__main__":

  # now initilise ros
  rospy.init_node("ft_sensor_node")
  rospy.loginfo("The F/T sensor node has now started")

  # # create a publisher for force torque data
  # forces_pub = rospy.Publisher("/gripper/real/ft/forces", FloatXYZ, queue_size=10)
  # torques_pub = rospy.Publisher("/gripper/real/ft/torques", FloatXYZ, queue_size=10)

  p = Popen('/home/luke/ftdriver/Linux/bin/driverSensor', stdout=PIPE,
            stderr=STDOUT, shell=True)

  # while p.poll() is None:
  while True:

    line = p.stdout.readline()

    rospy.loginfo(line)