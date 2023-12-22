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

def demand_callback(data):
  """
  Receive a gripper ROS input demand and send it to the real gripper
  """

  # check for commands that override and cause ignoring of others
  if data.stop: 
    mygripper.send_message(type="stop")
    return
  if data.home: 
    mygripper.send_message(type="home")
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

  try:

    # establish connection with the gripper
    bt_port = "/dev/rfcomm0"
    usb_port = "/dev/ttyACM0"

    mygripper = Gripper()
    mygripper.connect(usb_port)

    # create output message
    output_msg = GripperOutput()

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

    # # for testing: publish rolling averages
    # gauge1avg_pub = rospy.Publisher("/gripper/real/gauge1avg", Float64, queue_size=10)
    # gauge2avg_pub = rospy.Publisher("/gripper/real/gauge2avg", Float64, queue_size=10)
    # gauge3avg_pub = rospy.Publisher("/gripper/real/gauge3avg", Float64, queue_size=10)
    # gauge4avg_pub = rospy.Publisher("/gripper/real/gauge4avg", Float64, queue_size=10)
    # i = 0
    # num_for_avg = 10
    # gauge1avg = np.zeros(num_for_avg)
    # gauge2avg = np.zeros(num_for_avg)
    # gauge3avg = np.zeros(num_for_avg)
    # gauge4avg = np.zeros(num_for_avg)

    gripper_message_rate = 20

    rate = rospy.Rate(gripper_message_rate) # 10Hz
 
    while not rospy.is_shutdown():

      if not gripper_initialised and mygripper.connected:

        mygripper.send_message(type="resume")
        mygripper.send_message(type="home")
        mygripper.send_message(type="debug_on")

        mygripper.command.x = 0.2
        mygripper.send_message(type="change_timed_action")
        mygripper.command.x = 0.0 # was 0.3 for tests
        mygripper.send_message(type="change_timed_action_early_pub")

        mygripper.command.x = 100
        mygripper.send_message(type="set_gauge_hz")
        mygripper.command.x = gripper_message_rate
        mygripper.send_message(type="set_publish_hz")
        mygripper.command.x = 20
        mygripper.send_message(type="set_serial_hz")
        mygripper.command.x = 100000
        mygripper.send_message(type="set_motor_hz")

        mygripper.send_message(type="debug_off")

        # # set a lower speed
        # mygripper.command.x = 150
        # mygripper.command.y = 150
        # mygripper.command.z = 150
        # mygripper.send_message(type="set_speed")

        gripper_initialised = True

      # get the most recent state of the gripper
      state = mygripper.update_state()

      # mygripper.send_message("print")

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

          # # get averages
          # gauge1avg[i] = state.gauge1_data
          # gauge2avg[i] = state.gauge2_data
          # gauge3avg[i] = state.gauge3_data
          # gauge4avg[i] = state.gauge4_data
          # i += 1
          # if i == num_for_avg:
          #   gauge1avg_pub.publish(np.mean(gauge1avg))
          #   gauge2avg_pub.publish(np.mean(gauge2avg))
          #   gauge3avg_pub.publish(np.mean(gauge3avg))
          #   gauge4avg_pub.publish(np.mean(gauge4avg))
          #   i = 0

      # otherwise try to reconnect
      else: 
        gripper_initialised = False
        mygripper.connect(usb_port)

      rate.sleep()

  except rospy.ROSInterruptException:
    pass

  if mygripper.connected: mygripper.send_message("stop")

  rospy.logerr("gripper connection node has shut down")