#!/usr/bin/env python3

import rospy
import sys
import numpy as np

from std_srvs.srv import Empty
from gripper_msgs.msg import GripperState, GripperDemand
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings

# insert the mymujoco path for TrainDQN file
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
device = "cuda" #none
dqn_log_level = 1
model = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)

# user settings defaults for this node
log_level = 2
action_delay = 1.0
move_gripper = False
move_panda = False

# global flags
new_demand = False
model_loaded = False
ready_for_new_action = False
continue_grasping = True

# these will be ROS publishers once the node starts up
norm_state_pub = None
norm_sensor_pub = None

# start position is zero (fingers should be 10mm above the ground)
panda_z_height = 0.0

def data_callback(state):
  """
  Receives new state data for the gripper
  """

  # we cannot start saving data until the model class is ready (else silent crash)
  if not model_loaded: return

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

  if log_level > 1: rospy.loginfo("generating a new action")

  obs = model.env.mj.get_real_observation()
  obs = model.to_torch(obs)
  
  # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
  action = model.select_action(obs, decay_num=1, test=True)

  if log_level > 1: rospy.loginfo(f"Action code is: {action.item()}")

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

  rospy.loginfo(f"New panda target is {target_z * 1000} mm")

  # hardcoded safety checks
  min = -10e-3
  max = 30e-3
  if target_z < min:
    rospy.logwarn(f"panda z target of {target_z} is below the minimum of {min}")
    return
  if target_z > max:
    rospy.logwarn(f"panda z target of {target_z} is above the maximum of {max}")
    return

  # define duration in seconds
  duration = 1.0

  franka.move("relative", T, duration)

  # update with the new target position
  global panda_z_height
  panda_z_height = target_z

  return

def execute_grasping_callback(request=None):
  """
  Service callback to complete a dqn grasping task
  """

  if log_level > 0: rospy.loginfo("Entered execute_grasping_callback()")

  global ready_for_new_action
  global action_delay
  global continue_grasping
  continue_grasping = True

  reset_all()

  rate = rospy.Rate(20)

  while not rospy.is_shutdown():

    if ready_for_new_action and model_loaded:

      # evaluate the network and get a new action
      new_target_state, for_franka = generate_action()

      # do we delay before performing action (eg 0.5 seconds)
      if action_delay is not None:
        if log_level > 1: 
          rospy.loginfo(f"Sleeping before action execution for {action_delay} seconds")
        rospy.sleep(action_delay)
        if log_level > 1: rospy.loginfo("Finished sleeping")

      # has the grasping task been cancelled
      if not continue_grasping: break

      # if the action is for the gripper
      if not for_franka:

        if move_gripper is False:
          if log_level > 1: rospy.loginfo(f"Gripper action ignored as move_gripper=False")
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
      elif for_franka:

        if move_panda is False:
          if log_level > 1: rospy.loginfo(f"Panda action ignored as move_panda=False")
          continue

        panda_target = -new_target_state[3] # negative/positive flipped
        if log_level > 1: rospy.loginfo(f"dqn is sending a panda control signal to z = {panda_target}")
        move_panda_z_abs(franka_instance, panda_target)

        # move panda is blocking, so we know we can now have a new action
        ready_for_new_action = True

    rate.sleep()

  rospy.loginfo("Leaving execute_grasping_callback()")

  return []

def cancel_grasping_callback(request=None):
  """
  Callback to end a grasping task, does not stop immediately
  """

  rospy.loginfo("Cancelling grasping now")

  global continue_grasping
  continue_grasping = False

  return []

def reset_all(request=None):
  """
  Reset the gripper and panda
  """

  global panda_z_height

  # reset and prepare environment
  model.env.reset()

  # create a homing request for the gripper
  homing_demand = GripperDemand()
  homing_demand.home = True

  # HERE WE SHOULD RESET PANDA POSITION BUT THIS IS NOT YET DONE
  rospy.logwarn("panda position should be reset but is NOT currently, panda_z_height set to 0")
  panda_z_height = 0.0

  if log_level > 0: rospy.loginfo("dqn node is publishing a homing gripper demand")
  demand_pub.publish(homing_demand)

  return []

def connect_panda(request=None):
  """
  Connect to the franka panda
  """

  global franka_instance
  global move_panda

  try:

    if log_level > 0: rospy.loginfo("Trying to begin panda connection")
    sys.path.insert(0, "/home/luke/franka_interface/build")
    import pyfranka_interface

    # create franka controller instance
    franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)
    move_panda = True
    if log_level > 0: rospy.loginfo("Panda connection started successfully")

  except Exception as e:
    rospy.logerr(e)
    rospy.logerr("Failed to start panda conenction")
    move_panda = False

def load_model(request):
  """
  Load a dqn model
  """
  
  global model_loaded
  global model

  # apply defaults
  if request.folderpath == "":
    request.folderpath = "/home/luke/mymujoco/rl/models/dqn/"
  if request.run_id == 0:
    request.run_id = None

  # construct paths
  pathtofolder = request.folderpath + "/" + request.group_name + "/"
  foldername = request.run_name

  try:

    if log_level > 0: rospy.loginfo(f"Preparing to load model now in dqn node: {pathtofolder}/{foldername}")
    model.load(id=request.run_id, folderpath=pathtofolder, foldername=foldername)
    model_loaded = True
    if log_level > 0: rospy.loginfo("Model loaded successfully")

    return True

  except Exception as e:

    rospy.logerr(e)
    rospy.logerr("Failed to load model in dqn node")

    return False

def apply_settings(request):
  """
  Override existing settings with that from request - beware that all settings are
  overwritten, so if you ignore a setting in the service request it will now be
  set to the default value from initilisation (so False or 0)
  """

  global log_level
  global action_delay
  global move_gripper
  global move_panda
  global franka_instance

  log_level = request.log_level
  action_delay = request.action_delay
  move_gripper = request.move_gripper
  move_panda = request.move_panda

  # if starting or ending a panda connection
  if move_panda: connect_panda()
  else: franka_instance = None

  return []

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("dqn_node")
  rospy.loginfo("dqn node main has now started")

  # get input parameters - are we allowing robot movement?
  move_gripper = rospy.get_param("/gripper/dqn/move_gripper")
  move_panda = rospy.get_param("/gripper/dqn/move_panda")
  rospy.loginfo(f"move gripper is {move_gripper}")
  rospy.loginfo(f"move panda is {move_panda}")

  # do we need to import the franka control library
  if move_panda: connect_panda()
    
  # subscriber for gripper data and publisher to send gripper commands
  rospy.Subscriber("/gripper/real/state", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher("/gripper/dqn/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher("/gripper/dqn/sensor", NormalisedSensor, queue_size=10)

  # define the model to load and then try to load it
  load = LoadModel()
  load.folderpath = "/home/luke/mymujoco/rl/models/dqn/"
  load.group_name = "07-11-22"
  load.run_name = "luke-PC_17:39_A8"
  load.run_id = None
  load_model(load)

  # uncomment for more debug information
  # model.env.log_level = 2
  # model.env.mj.set.debug = True

  # begin services for this node
  rospy.Service('/gripper/dqn/start', Empty, execute_grasping_callback)
  rospy.Service('/gripper/dqn/stop', Empty, cancel_grasping_callback)
  rospy.Service('/gripper/dqn/reset', Empty, reset_all)
  rospy.Service("/gripper/dqn/connect_panda", Empty, connect_panda)
  rospy.Service("/gripper/dqn/load_model", LoadModel, load_model)
  rospy.Service("/gripper/dqn/apply_settings", ApplySettings, apply_settings)

  rospy.spin() # and wait for service requests
  