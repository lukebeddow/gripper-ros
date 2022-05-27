#!/usr/bin/env python

import rospy
import serial
from math import sin
from gripper_class import Gripper
from gripper_virtual_node.msg import gripper_msg
from gripper_virtual_node.msg import gripper_state

from gripper_msgs.msg import demand
from gripper_msgs.msg import status


from std_msgs.msg import Float64
from std_msgs.msg import Bool

def demand_callback(data):
  """callback function for when command is received for the gripper"""

  use_units = None

  # check for special message cases
  if data.stop: command_type = "stop"
  elif data.resume: command_type = "resume"
  elif data.home: command_type = "home"
  elif data.power_saving_off: command_type = "power_saving_off"
  elif data.power_saving_on: command_type = "power_saving_on"

  # otherwise we are sending a command
  else: 
    command_type = "command"
    # check what units have been requested
    if data.y_is_angle:
      if data.use_deg:
        if not data.use_mm:
          rospy.logwarn("Gripper command cannot have units 'metres' and 'degrees'"
            ", units have been overridden to 'millimeters' and 'degrees'")
        use_units = "mm_deg"
      else:
        if data.use_mm:
          rospy.logwarn("Gripper command cannot have units 'millimetres' and 'radians'"
            ", units have been overridden to 'metres' and 'radians'")
        use_units = "m_rad"
    elif data.use_mm:
      use_units = "mm"
    else:
      use_units = "m"

  # input command data
  mygripper.command.x = data.gripper.x
  mygripper.command.y = data.gripper.y
  mygripper.command.z = data.gripper.z

  log_str = "Sending gripper command with units %s of (x, y, z): (%.2f, %.2f, %.2f)" % (
    use_units, mygripper.command.x, mygripper.command.y, mygripper.command.z
  )

  rospy.loginfo(log_str)

  # send the command
  mygripper.send_message(type=command_type, units=use_units)

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
    rospy.Subscriber("/gripper/demand", demand, demand_callback)
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