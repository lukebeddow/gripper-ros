#!/usr/bin/env python3

import rospy
import sys
import numpy as np

from gripper_msgs.msg import GripperInput, GripperState, GripperDemand
from gripper_msgs.msg import NormalisedState, NormalisedSensor

# insert the mymujoco path
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
device = "cpu" #none
model = TrainDQN(use_wandb=False, no_plot=True, log_level=1, device=device)

# globals
log_level = 0

gauge_actual_pub = None
prev_g1 = 0
prev_g2 = 0
prev_g3 = 0

new_demand = False
model_loaded = False
ready_for_new_action = False

demand = GripperInput()

# these will be ROS publishers once the node starts up
norm_state_pub = None
norm_sensor_pub = None

def data_callback(state):
  """
  Receives new state data for the gripper
  """

  # we cannot start saving data until the model class is ready (else silent crash)
  if not model_loaded: return

  panda_z_height = 0

  # create vectors of observation data, order is very important!
  state_vec = [
    state.pose.x, state.pose.y, state.pose.z, panda_z_height
  ]

  sensor_vec = [
    state.sensor.gauge1, state.sensor.gauge2, state.sensor.gauge3, 
    state.sensor.gauge4,
    state.ftdata.force.z
  ]
  
  timestamp = 0 # not using this for now

  # input the data
  model.env.mj.input_real_data(state_vec, sensor_vec, timestamp)

  # for testing, get the normalised data values as the network will see them
  unnormalise = False # we want normalised values
  [g1, g2, g3] = model.env.mj.get_bend_gauge_readings(unnormalise)
  p = model.env.mj.get_palm_reading(unnormalise)
  wZ = model.env.mj.get_wrist_reading(unnormalise)
  [gx, gy, gz, bz] = model.env.mj.get_state_readings(unnormalise)

  # publish the dqn network normalised input values
  global norm_state_pub, norm_sensor_pub
  norm_state = NormalisedState()
  norm_sensor = NormalisedSensor()

  norm_state.gripper_x = gx
  norm_state.gripper_y = gy
  norm_state.gripper_z = gz
  norm_state.base_z = bz

  norm_sensor.gauge1 = g1
  norm_sensor.gauge2 = g2 
  norm_sensor.gauge3 = g3 
  norm_sensor.palm = p
  norm_sensor.wrist_z = wZ

  norm_state_pub.publish(norm_state)
  norm_sensor_pub.publish(norm_sensor)

  # once an action is completed, we need a new one
  if state.is_target_reached:

    global ready_for_new_action
    ready_for_new_action = True

def generate_action():
  """
  Get a new gripper state
  """

  if log_level > 0: rospy.loginfo("generating a new action")

  obs = model.env.mj.get_real_observation()
  obs = model.to_torch(obs)
  
  # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
  action = model.select_action(obs, decay_num=1, test=True)

  if log_level > 0: rospy.loginfo(f"Action code is: {action.item()}")

  # apply the action and get the new target state (vector)
  new_target_state = model.env.mj.set_action(action.item())

  # determine if this action is for the gripper or panda
  if model.env.mj.last_action_gripper(): for_franka = False
  elif model.env.mj.last_action_panda(): for_franka = True
  else:
    raise RuntimeError("last action not on gripper or on panda")

  return new_target_state, for_franka

def move_panda_z_abs(franka, target_z):
  """
  Move the panda to a new z position with cartesian motion using Valerio's library,
  an instance of which is passed as 'franka'
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

  # get parameters - are we allowing robot movement?
  move_gripper = rospy.get_param("/gripper/dqn/move_gripper")
  move_panda = rospy.get_param("/gripper/dqn/move_panda")
  rospy.loginfo(f"move gripper is {move_gripper}")
  rospy.loginfo(f"move panda is {move_panda}")

  # THIS CODE AND PANDA MOVEMENTS HAVE NOT BEEN VERIFIED TO WORK
  # do we need to import the franka control library
  if move_panda:
    
    sys.path.insert(0, "/home/luke/franka_interface/build")
    import pyfranka_interface

    # create franka controller instance
    franka_instance = pyfranka_interface.Robot_("172.16.0.2")
  
  # subscriber for gripper data and publisher to send gripper commands
  rospy.Subscriber("/gripper/real/stattargete", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher("/gripper/dqn/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher("/gripper/dqn/sensor", NormalisedSensor, queue_size=10)

  # load the trained model file
  try:
    if log_level > 0: rospy.loginfo("Preparing to load model now in dqn node")
    folderpath = "/home/luke/mymujoco/rl/models/dqn/baselines-oct/"
    foldername = "sensor_2_thickness_0.9"
    model.load(id=None, folderpath=folderpath, foldername=foldername)
    model_loaded = True
    if log_level > 0: rospy.loginfo("Model loaded successfully")

  except Exception as e:
    print(e)
    if log_level > 0: rospy.logerr("Failed to load model in dqn node")

  # uncomment for more debug information
  # model.env.log_level = 2
  # model.env.mj.set.debug = True

  rate = rospy.Rate(20)

  while not rospy.is_shutdown():

    if ready_for_new_action and model_loaded:

      # evaluate the network and get a new action
      new_target_state, for_franka = generate_action()

      # do we delay before performing action (eg 0.5 seconds)
      delay = None
      if delay is not None:
        rospy.loginfo(f"Sleeping before action execution for {delay} seconds")
        rospy.sleep(delay)
        rospy.loginfo("Finished sleeping")

      # if the action is for the gripper
      if for_franka == False:

        if move_gripper is False:
          if log_level > 0: rospy.loginfo(f"Gripper action ignored as move_gripper=False")
          continue

        new_demand = GripperDemand()
        new_demand.state.pose.x = new_target_state[0]
        new_demand.state.pose.y = new_target_state[1]
        new_demand.state.pose.z = new_target_state[2]

        if log_level > 0: rospy.loginfo("dqn node is publishing a new gripper demand")
        demand_pub.publish(new_demand)

        # data callback will let us know when the gripper demand is fulfilled
        ready_for_new_action = False

      # if the action is for the panda
      elif for_franka == True:

        if move_panda is False:
          if log_level > 0: rospy.loginfo(f"Panda action ignored as move_panda=False")
          continue

        panda_target = new_target_state[3]
        if log_level > 0: rospy.loginfo(f"dqn is sending a panda control signal to z = {panda_target}")
        move_panda_z_abs(franka_instance, panda_target)

        # move panda is blocking, so we know we can now have a new action
        ready_for_new_action = True

    rate.sleep()