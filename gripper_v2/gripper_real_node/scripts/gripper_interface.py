#!/home/luke/pyenv/py38_ros/bin/python

import rospy
from gripper_class import Gripper
from gripper_msgs.msg import GripperInput
from gripper_msgs.msg import GripperOutput
from gripper_msgs.msg import GripperRequest
from std_msgs.msg import Float64
from std_msgs.msg import Bool

# # for testing rolling averages
# import numpy as np

def command_line_callback(data):
  """
  Gripper demand message from the user on the command line
  """

  mygripper.command.x = data.x
  mygripper.command.y = data.y
  mygripper.command.z = data.z
  mygripper.send_message(type=data.message_type)

def other_demand_callback(data):
  """
  Gripper demand message for using a second gripper (ie other gripper)
  """

  othergripper.command.x = data.x
  othergripper.command.y = data.y
  othergripper.command.z = data.z
  othergripper.send_message(type=data.message_type)

def demand_callback(data):
  """
  Receive a gripper ROS input demand and send it to the real gripper
  """

  # check for commands that override and cause ignoring of others
  if data.stop: 
    mygripper.send_message(type="stop")
    return
  if data.home: 
    print(f"Sending homing message of type: {home_type}")
    mygripper.send_message(type=home_type)
    print(f"Message sent")
    return
    
  if data.print_debug:
    rospy.loginfo("gripper_interface.py: sending request for debug info to gripper")
    mygripper.send_message(type="print")
    return

  if data.resume: mygripper.send_message(type="resume")
  if data.power_saving_on: mygripper.send_message(type="power_saving_on")
  if data.power_saving_off: mygripper.send_message(type="power_saving_off")

  if not data.ignore_xyz_command:

    # input command data
    use_units = "m"
    mygripper.command.x = data.x_m
    mygripper.command.y = data.y_m
    mygripper.command.z = data.z_m

    log_str = "Sending gripper command with units %s of (x, y, z): (%.4f, %.4f, %.4f)" % (
      use_units, mygripper.command.x, mygripper.command.y, mygripper.command.z
    )

    rospy.loginfo(log_str)

    mygripper.send_message(type="timed_command", units=use_units)

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
  output_msg.gauge4 = state.gauge4_data

  return output_msg

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("gripper_real_publisher")

  gripper_initialised = False

  use_both_grippers = True

  try:

    # establish connection with the gripper
    bt_port = "/dev/rfcomm0"
    usb_port = "/dev/ttyACM1"

    if use_both_grippers:
      othergripper = Gripper()
      othergripper.connect("/dev/ttyACM0")

    mygripper = Gripper()
    mygripper.connect(usb_port)

    # add code to figure out which gripper is which? Could have a message ping that
    # identifies each gripper?

    # create raw data publishers
    connected_pub = rospy.Publisher("/gripper/real/connected", Bool, queue_size=10)
    gauge1_pub = rospy.Publisher("/gripper/real/gauge1", Float64, queue_size=10)
    gauge2_pub = rospy.Publisher("/gripper/real/gauge2", Float64, queue_size=10)
    gauge3_pub = rospy.Publisher("/gripper/real/gauge3", Float64, queue_size=10)
    gauge4_pub = rospy.Publisher("/gripper/real/gauge4", Float64, queue_size=10)

    # create data transfer input/output
    rospy.Subscriber("/gripper/real/input", GripperInput, demand_callback)
    rospy.Subscriber("/gripper/request", GripperRequest, command_line_callback)
    state_pub = rospy.Publisher("/gripper/real/output", GripperOutput, queue_size=10)

    # publishers and subscribers for using both grippers
    if use_both_grippers:
      rospy.Subscriber("/gripper/other/request", GripperRequest, other_demand_callback)
      other_pub = rospy.Publisher("/gripper/other/output", GripperOutput, queue_size=10)

    # do we stop all message sending/receiving during homing, to reduce noises
    use_homing_blocking = True
    if use_homing_blocking:
      home_type = "home_blocking"
    else:
      home_type = "home"

    # gripper key settings
    gripper_publish_hz = 20
    gripper_serial_hz = 20
    gripper_gauge_hz = 80
    gripper_motor_hz = 100000
    gripper_xy_speed_rpm = 200
    gripper_z_speed_rpm = 400
    gripper_timed_action_s = 0.3
    gripper_timed_action_early_pub_s = 0.0
    gripper_xy_rpm = 200
    gripper_z_rpm = 400

    rate = rospy.Rate(gripper_publish_hz) # 10Hz
 
    while not rospy.is_shutdown():

      if not gripper_initialised and mygripper.connected:

        # turn on debugging so changes are echoed in terminal
        mygripper.send_message(type="debug_on")

        # input settings
        mygripper.send_message(type="resume")
        mygripper.send_message(type=home_type)
        mygripper.send_message(type="power_saving_on")
        mygripper.send_message(type="change_timed_action", value=gripper_timed_action_s)
        mygripper.send_message(type="change_timed_action_early_pub", value=gripper_timed_action_early_pub_s)
        mygripper.send_message(type="set_publish_hz", value=gripper_publish_hz)
        mygripper.send_message(type="set_serial_hz", value=gripper_serial_hz)
        mygripper.send_message(type="set_gauge_hz", value=gripper_gauge_hz)
        mygripper.send_message(type="set_motor_hz", value=gripper_motor_hz)

        # set the speed
        mygripper.command.x = gripper_xy_rpm
        mygripper.command.y = gripper_xy_rpm
        mygripper.command.z = gripper_z_rpm
        mygripper.send_message(type="set_speed")

        mygripper.send_message(type="debug_off")
        gripper_initialised = True

      # get the most recent state of the gripper
      state = mygripper.update_state()

      # check if the connection is live yet
      connected_pub.publish(mygripper.connected)

      if mygripper.connected:

        if mygripper.first_message_received:

          # fill in the gripper output message
          output_msg = state_to_msg(state)

          # publish data
          state_pub.publish(output_msg)

          # for visualisation ONLY
          gauge1_pub.publish(state.gauge1_data)
          gauge2_pub.publish(state.gauge2_data)
          gauge3_pub.publish(state.gauge3_data)
          gauge4_pub.publish(state.gauge4_data)

      # otherwise try to reconnect
      else: 
        gripper_initialised = False
        mygripper.connect(usb_port)

      if use_both_grippers:

        otherstate = othergripper.update_state()
        
        if othergripper.connected:
          if othergripper.first_message_received:
            other_msg = state_to_msg(otherstate)
            other_pub.publish(other_msg)
        else:
          rospy.logwarn("use_both_grippers = True, but NOT connected")

      rate.sleep()

  except rospy.ROSInterruptException:
    pass

  if mygripper.connected: mygripper.send_message("stop")

  rospy.logerr("gripper connection node has shut down")