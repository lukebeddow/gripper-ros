#!/usr/bin/env python

import rospy
from gripper_msgs.srv import move_gripper, move_gripperRequest, move_gripperResponse
from std_msgs.msg import String

def callback(data):
  print("entered callback function with data", data.data)
  if data.data == "s":
    # create service request message
    msg = move_gripperRequest()
    msg.nudge.arm.x = 0.001
    # msg.nudge.arm.y = -0.1
    # msg.nudge.arm.z = 0.0
    msg.nudge.arm.roll = 0.1
    msg.nudge.arm.pitch = 0
    msg.nudge.arm.yaw = 0
    # msg.nudge.gripper.x = -10
    # msg.nudge.gripper.th = -10
    # msg.nudge.gripper.z = 30

    resp = move_gripperResponse

    try:
      resp = move_gripper_srv(msg)
    except rospy.ServiceException as exc:
      print("Service did not proces request: " + str(exc))

if __name__ == "__main__":

  rospy.init_node("testing_node")

  rospy.Subscriber("/PluginTest/string", String, callback)

  rospy.wait_for_service("/move_gripper")
  print("Service found")
  move_gripper_srv = rospy.ServiceProxy("/move_gripper", move_gripper)

  

  rospy.spin()

  