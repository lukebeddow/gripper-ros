#!/usr/bin/env python

import rospy

from gripper_class import Gripper

from gripper_msgs.msg import GripperInput
from gripper_msgs.msg import GripperOutput
from std_msgs.msg import Float64
from std_msgs.msg import Bool

def demand_callback(data):
  """
  Receive a gripper ROS input demand and send it to the real gripper
  """

  log_str = ("gripper input: home = %d, stop = %d, resume = %d, power_saving_on = %d, power_saving_off = %d, ignore_xyz_command = %d" % 
    (data.home, data.stop, data.resume, data.power_saving_on, data.power_saving_off, data.ignore_xyz_command))

  rospy.loginfo(log_str)

  # check demand information in order of importance
  if data.stop: mygripper.send_message(type="stop")
  if data.resume: mygripper.send_message(type="resume")
  if data.power_saving_on: mygripper.send_message(type="power_saving_on")
  if data.power_saving_off: mygripper.send_message(type="power_saving_off")
  if data.home: mygripper.send_message(type="home")

  if not data.ignore_xyz_command:

    # input command data
    use_units = "m"
    mygripper.command.x = data.x_m
    mygripper.command.y = data.y_m
    mygripper.command.z = data.z_m

    log_str = "Sending gripper command with units %s of (x, y, z): (%.2f, %.2f, %.2f)" % (
      use_units, mygripper.command.x, mygripper.command.y, mygripper.command.z
    )

    rospy.loginfo(log_str)

    mygripper.send_message(type="command", units=use_units)

def state_to_msg(state):
  """
  Convert the gripper class state to a ROS GripperOutput message
  """

  output_msg = GripperOutput()

  output_msg.is_target_reached = state.is_target_reached
  output_msg.motor_x_m = state.x_mm * 1e-3
  output_msg.motor_y_m = state.y_mm * 1e-3
  output_msg.motor_z_m = state.z_mm * 1e-3
  output_msg.gauge1 = state.gauge1_data
  output_msg.gauge2 = state.gauge2_data
  output_msg.gauge3 = state.gauge3_data

  return output_msg

if __name__ == "__main__":

  try:

    # establish connection with the gripper
    com_port = "/dev/rfcomm0"
    mygripper = Gripper()
    mygripper.connect(com_port)

    # now initilise ros
    rospy.init_node("gripper_real_publisher")

    # create output message
    output_msg = GripperOutput()

    # create raw data publishers
    connected_pub = rospy.Publisher("/gripper/real/connected", Bool, queue_size=10)
    gauge1_pub = rospy.Publisher("/gripper/real/gauge1", Float64, queue_size=10)
    gauge2_pub = rospy.Publisher("/gripper/real/gauge2", Float64, queue_size=10)
    gauge3_pub = rospy.Publisher("/gripper/real/gauge3", Float64, queue_size=10)

    # create data transfer input/output
    rospy.Subscriber("/gripper/real/input", GripperInput, demand_callback)
    state_pub = rospy.Publisher("/gripper/real/output", GripperOutput, queue_size=10)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

      # get the most recent state of the gripper
      state = mygripper.update_state()

      # check if the connection is live yet
      connected_pub.publish(mygripper.connected)

      if mygripper.connected:

        # fill in the gripper output message
        output_msg = state_to_msg(state)

        # publish data
        state_pub.publish(output_msg)
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