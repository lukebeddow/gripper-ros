#!/usr/bin/env python3

import rospy
import sys
from gripper_msgs.srv import ControlRequest, ControlRequestResponse
import networks
import numpy as np

from gripper_msgs.msg import GripperInput, GripperOutput, GripperState, GripperDemand, MotorState
from gripper_msgs.msg import NormalisedState, NormalisedSensor

# insert the mymujoco path
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
device = "cuda" #none
model = TrainDQN(use_wandb=False, no_plot=True, log_level=1, device=device)

# insert the franka_interface path
sys.path.insert(0, "/home/luke/franka_interface/build")

# import pyfranka_interface

# # create franka controller instance
# franka = pyfranka_interface.Robot_("172.16.0.2")

# globals
gauge_actual_pub = None
prev_g1 = 0
prev_g2 = 0
prev_g3 = 0

new_demand = False
demand = GripperInput()

ready_for_new_action = False

new_data_from_gripper = False
normalised_data = []

def data_callback(state):
  """
  Receives new state data for the gripper
  """

  state_vec = [
    state.pose.x, state.pose.y, state.pose.z
  ]

  sensor_vec = [
    state.sensor.gauge1, state.sensor.gauge2, state.sensor.gauge3, state.sensor.gauge4
  ]
  
  timestamp = 0 # not using this for now

  global normalised_data
  global new_data_from_gripper
  normalised_data = model.env.mj.input_real_data(state_vec, sensor_vec, timestamp)
  new_data_from_gripper = True

  if state.is_target_reached:

    global ready_for_new_action
    ready_for_new_action = True

    # # for testing - visualise the actual gauge network inputs
    # vec = model.env.mj.get_finger_gauge_data()
    # global gauge_actual_pub
    # mymsg = MotorState()
    # mymsg.x = vec[0]
    # mymsg.y = vec[1]
    # mymsg.z = vec[2]
    # gauge_actual_pub.publish(mymsg)

def generate_action():
  """
  Get a new gripper state
  """

  rospy.loginfo("generating a new action")

  obs = model.env.mj.get_real_observation()
  obs = model.to_torch(obs)
  
  # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
  action = model.select_action(obs, decay_num=1, test=True)

  rospy.loginfo(f"Action code is: {action.item()}")

  # set the service response with the new position this action results in
  new_target_state = model.env.mj.set_action(action.item())

  # determine if this action is for the gripper or panda
  if model.env.mj.last_action_gripper(): for_franka = False
  elif model.env.mj.last_action_panda(): for_franka = True
  else:
    raise RuntimeError("last action not on gripper or on panda")

  # rospy.loginfo("Sleeping now")
  # rospy.sleep(0.5) # Sleeps for 0.5 sec
  # rospy.loginfo("Finished sleeping")

  return new_target_state, for_franka

def move_panda_z_abs(target_z):
  """
  Move the panda to a new z position with cartesian motion using Valerio's library
  """

  # create identity matrix (must be floats!)
  T = np.array(
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]],
     dtype=np.float
  )

  # insert z target into matrix
  T[2,3] = target_z

  # define duration in seconds
  duration = 0.4

  franka.move("relative", T, duration)

  return

if __name__ == "__main__":

  # now initilise ros
  rospy.init_node("dqn_node")
  rospy.loginfo("dqn node main has now started")

  # create service responder
  # rospy.Service("/gripper/control/dqn", ControlRequest, srv_callback)
  
  rospy.Subscriber("/gripper/real/state", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher("/gripper/dqn/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher("/gripper/dqn/sensor", NormalisedSensor, queue_size=10)

  # gauge_actual_pub = rospy.Publisher("/gripper/dqn/gauges", MotorState, queue_size=10)
  # rospy.Subscriber("/gripper/real/output", GripperOutput, state_callback)
  # demand_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)

  # load the file that is local
  folderpath = "/home/luke/mymujoco/rl/models/dqn/baselines-oct/"
  foldername = "sensor_2_thickness_0.9"
  model.load(id=None, folderpath=folderpath, foldername=foldername)

  model.env.mj.set.debug = True

  rate = rospy.Rate(20)

  while not rospy.is_shutdown():

    perform_actions = False

    if perform_actions:
      if ready_for_new_action: #and not no_motion:

        new_target_state, for_franka = generate_action()

        # if the target is for the gripper
        if for_franka == False:

          new_demand = GripperDemand()
          new_demand.state.pose.x = new_target_state[0]
          new_demand.state.pose.y = new_target_state[1]
          new_demand.state.pose.z = new_target_state[2]

          rospy.loginfo("dqn node is publishing a new gripper demand")
          demand_pub.publish(new_demand)

        # if the target is for the panda
        elif for_franka == True:

          rospy.loginfo("PANDA action: ignored")
          continue

          panda_target = new_target_state[3]
          rospy.loginfo(f"dqn is sending a panda control signal to z = {panda_target}")
          move_panda_z_abs(panda_target)

        ready_for_new_action = False

    if new_data_from_gripper:

      norm_state = NormalisedState()
      norm_sensor = NormalisedSensor()

      i = 0

      # fill in state message
      if model.env.mj.set.motor_state_sensor.in_use:
        norm_state.gripper_x = normalised_data[i]; i += 1
        norm_state.gripper_y = normalised_data[i]; i += 1
        norm_state.gripper_z = normalised_data[i]; i += 1
      if model.env.mj.set.base_state_sensor.in_use:
        norm_state.base_z = normalised_data[i]; i += 1

      # fill in sensor message
      if model.env.mj.set.bending_gauge.in_use:
        norm_sensor.gauge1 = normalised_data[i]; i += 1
        norm_sensor.gauge2 = normalised_data[i]; i += 1
        norm_sensor.gauge3 = normalised_data[i]; i += 1
      if model.env.mj.set.palm_sensor.in_use:
        norm_sensor.palm = normalised_data[i]; i += 1
      if model.env.mj.set.wrist_sensor_Z.in_use:
        norm_sensor.wrist_z = normalised_data[i]; i += 1

      norm_state_pub.publish(norm_state)
      norm_sensor_pub.publish(norm_sensor)

      new_data_from_gripper = False

    rate.sleep()