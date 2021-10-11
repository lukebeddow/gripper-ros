#!/usr/bin/env python

import rospy
import serial
from math import sin
from gripper_class import Gripper
from gripper_virtual_node.msg import gripper_msg
from gripper_virtual_node.msg import gripper_state
from std_msgs.msg import Float64
from std_msgs.msg import Bool

def callback(data):
  """callback function for when command is received for the gripper"""

  mygripper.command.radius = data.x * 1000.0
  mygripper.command.angle = data.th * (-180.0 / 3.141592)
  mygripper.command.palm = data.z * 1000.0

  mygripper.send_command()



if __name__ == "__main__":
  try:
    # establish connection with the gripper
    com_port = "/dev/rfcomm0"
    mygripper = Gripper()
    mygripper.connect(com_port)

    # now initilise ros
    rospy.init_node("gripper_real_publisher")
    msg = gripper_msg()
    cmd = gripper_state()
    real_pub = rospy.Publisher("/gripper/real/connected", Bool, queue_size=10)
    state_pub = rospy.Publisher("/gripper/real/status", gripper_msg, queue_size=10)
    gauge1_pub = rospy.Publisher("/gripper/real/gauge1", Float64, queue_size=10)
    gauge2_pub = rospy.Publisher("/gripper/real/gauge2", Float64, queue_size=10)
    gauge3_pub = rospy.Publisher("/gripper/real/gauge3", Float64, queue_size=10)
    rospy.Subscriber("/gripper/demand", gripper_state, callback)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
      # check if the connection is live yet
      real_pub.publish(mygripper.connected)

      # get the most recent state of the gripper
      mygripper.update_state()
      state = mygripper.get_state()

      # extract the finger positions
      msg.x = state.x_mm
      msg.y = state.y_mm
      msg.z = state.z_mm
      msg.th = sin((msg.x - msg.y) / 35.0)

      # publish data
      state_pub.publish(msg)
      gauge1_pub.publish(state.gauge1_data)
      gauge2_pub.publish(state.gauge2_data)
      gauge3_pub.publish(state.gauge3_data)

      rate.sleep()

  except rospy.ROSInterruptException:
    pass

  rospy.logerr("gripper connection node has shut down")
  # except:
  #   # keep broadcasting that the gripper is disconnected for 10 seconds
  #   rospy.logerr("gripper has been disconnected, connection lost")
  #   exception_rate = rospy.Rate(10)
  #   for i in range(100):
  #     real_pub.publish(False)
  #     exception_rate.sleep()

  # except Gripper.GripperException:
    # rospy.logerr("Failed to connect to gripper via bluetooth, node shutting down")