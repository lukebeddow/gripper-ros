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

  # send in units of mm and radians
  mygripper.command.radius = data.x * 1000.0
  mygripper.command.angle = data.th * -1
  mygripper.command.palm = data.z * 1000.0

  mygripper.send_command()

  # global test

  # if test % 4 == 0:
  #   mygripper.send_message("power_saving_on")
  #   print("power saving ON")
  # elif test % 4 == 1:
  #   mygripper.send_message("power_saving_off")
  #   print("power saving OFF")
  # elif test % 4 == 2:
  #   mygripper.send_message("stop")
  #   print("gripper stopped")
  # elif test % 4 == 3:
  #   mygripper.send_message("resume")
  #   print("gripper resumed")

  # test += 1

# test = 0

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
      msg.is_target_reached = state.is_target_reached
      msg.x = state.x_mm# / 1000.0
      msg.y = state.y_mm# / 1000.0
      msg.z = state.z_mm# / 1000.0
      msg.th = sin((state.x_mm - state.y_mm) / 35.0)

      # publish data
      state_pub.publish(msg)
      gauge1_pub.publish(state.gauge1_data)
      gauge2_pub.publish(state.gauge2_data)
      gauge3_pub.publish(state.gauge3_data)

      rate.sleep()

  except rospy.ROSInterruptException:
    pass

  mygripper.send_message("stop")
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