#!/usr/bin/env python

import rospy

from gripper_class import Gripper

from gripper_msgs.msg import GripperInput
from gripper_msgs.msg import GripperOutput
from std_msgs.msg import Float64
from std_msgs.msg import Bool

# for testing
import numpy as np

def demand_callback(data):
  """
  Receive a gripper ROS input demand and send it to the real gripper
  """

  # # MORE TESTING - override
  # if data.override != "":
  #   log_str = "Overriding demand with %s and (x, y, z): (%.2f, %.2f, %.2f)" % (
  #     data.override, data.x_m, data.y_m, data.z_m
  #   )
  #   rospy.loginfo(log_str)
  #   mygripper.command.x = data.x_m
  #   mygripper.command.y = data.y_m
  #   mygripper.command.z = data.z_m
  #   mygripper.send_message(type=data.override)

  #   return

  # log_str = ("gripper input: home = %d, stop = %d, resume = %d, power_saving_on = %d, power_saving_off = %d, ignore_xyz_command = %d" % 
  #   (data.home, data.stop, data.resume, data.power_saving_on, data.power_saving_off, data.ignore_xyz_command))

  # rospy.loginfo(log_str)

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

    mygripper.send_message(type="command", units=use_units)

def scale_gauges(gauge1, gauge2, gauge3):
  """
  First attempt at scaling the gauges to [-1, 1] for nn input
  """

  """
  This scaling is very rough to put the gauges into SI units. A large deformation
  of the finger results in a value of -2.0 reading from the gauge. The wire stripper
  weight resulted in a reading of about 1.2. These would be 200g and 120g. The maximum
  the fingers could safely bend is likely about 300g, or +-3.0. Hence, we want to
  normalise the reading from -3.0 to +3.0.
  """
  gauge1 = (gauge1 + 0.70e6) * 1.258e-6
  gauge2 = (gauge2 - 0.60e6) * 1.258e-6
  gauge3 = (gauge3 - 0.56e6) * 1.258e-6
  gauge4 = (gauge4 - 0) * 1.0

  gauge1 = normalise_between(gauge1, -2, 2)
  gauge2 = normalise_between(gauge2, -2, 2)
  gauge3 = normalise_between(gauge3, -2, 2)
  gauge4 = normalise_between(gauge4, -1e6, 1e6)

  return gauge1, gauge2, gauge3, gauge4

def state_to_msg(state):
  """
  Convert the gripper class state to a ROS GripperOutput message
  """

  output_msg = GripperOutput()

  # g1, g2, g3 = scale_gauges(state.gauge1_data, state.gauge2_data, state.gauge3_data)
  # output_msg.gauge1 = g1
  # output_msg.gauge2 = g2
  # output_msg.gauge3 = g3

  output_msg.is_target_reached = state.is_target_reached
  output_msg.motor_x_m = state.x_mm * 1e-3
  output_msg.motor_y_m = state.y_mm * 1e-3
  output_msg.motor_z_m = state.z_mm * 1e-3

  output_msg.gauge1 = state.gauge1_data
  output_msg.gauge2 = state.gauge2_data
  output_msg.gauge3 = state.gauge3_data
  output_msg.gauge4 = state.gauge4_data

  return output_msg

def normalise_between(value, min, max):
  """
  Normalises a value into [-1, 1]
  """

  if value < min: return -1.0
  elif value > max: return 1.0
  else:
    return 2 * (value - min) / (max - min) - 1

if __name__ == "__main__":

  try:

    # establish connection with the gripper
    bt_port = "/dev/rfcomm0"
    usb_port = "/dev/ttyACM0"

    mygripper = Gripper()

    mygripper.connect(usb_port)
    mygripper.send_message(type="resume")

    # set a lower speed
    mygripper.command.x = 150
    mygripper.command.y = 150
    mygripper.command.z = 150
    mygripper.send_message(type="set_speed")

    # # TESTING - move closer
    # mygripper.command.x = 0.1
    # mygripper.command.y = 0.1
    # mygripper.command.z = 0.01
    # mygripper.send_message(type="command", units="m")

    # now initilise ros
    rospy.init_node("gripper_real_publisher")

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
    state_pub = rospy.Publisher("/gripper/real/output", GripperOutput, queue_size=10)

    # for testing: publish rolling averages
    gauge1avg_pub = rospy.Publisher("/gripper/real/gauge1avg", Float64, queue_size=10)
    gauge2avg_pub = rospy.Publisher("/gripper/real/gauge2avg", Float64, queue_size=10)
    gauge3avg_pub = rospy.Publisher("/gripper/real/gauge3avg", Float64, queue_size=10)
    gauge4avg_pub = rospy.Publisher("/gripper/real/gauge4avg", Float64, queue_size=10)
    i = 0
    num_for_avg = 10
    gauge1avg = np.zeros(num_for_avg)
    gauge2avg = np.zeros(num_for_avg)
    gauge3avg = np.zeros(num_for_avg)
    gauge4avg = np.zeros(num_for_avg)

    rate = rospy.Rate(10) # 10Hz
 
    while not rospy.is_shutdown():

      # get the most recent state of the gripper
      state = mygripper.update_state()

      # mygripper.send_message("print")

      # check if the connection is live yet
      connected_pub.publish(mygripper.connected)

      if mygripper.connected:

        # fill in the gripper output message
        output_msg = state_to_msg(state)

        # publish data
        state_pub.publish(output_msg)

        # for visualisation ONLY
        gauge1_pub.publish(state.gauge1_data)
        gauge2_pub.publish(state.gauge2_data)
        gauge3_pub.publish(state.gauge3_data)
        gauge4_pub.publish(state.gauge4_data)

        # get averages
        gauge1avg[i] = state.gauge1_data
        gauge2avg[i] = state.gauge2_data
        gauge3avg[i] = state.gauge3_data
        gauge4avg[i] = state.gauge4_data
        i += 1
        if i == num_for_avg:
          gauge1avg_pub.publish(np.mean(gauge1avg))
          gauge2avg_pub.publish(np.mean(gauge2avg))
          gauge3avg_pub.publish(np.mean(gauge3avg))
          gauge4avg_pub.publish(np.mean(gauge4avg))
          i = 0

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