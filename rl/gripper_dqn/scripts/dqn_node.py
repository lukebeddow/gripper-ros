#!/usr/bin/env python3

import rospy
import sys
import os
import numpy as np
from datetime import datetime

from std_srvs.srv import Empty, SetBool
from gripper_msgs.msg import GripperState, GripperDemand, GripperInput
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings, ResetPanda
from gripper_dqn.srv import StartTest, StartTrial, SaveTrial, LoadBaselineModel

try:
  global depth_camera_connected
  depth_camera_connected = True
  from capture_depth_image import get_depth_image
except: 
  print("DEPTH CAMERA NOT CONNECTED")
  depth_camera_connected = False
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

# import the test data structure class
from grasp_test_data import GraspTestData

# ----- essential settings and global variable declarations ----- #

# important user settings
log_level = 2                   # node log level, 0=disabled, 1=essential, 2=debug
action_delay = 0.1              # safety delay between action generation and publishing
panda_reset_height_mm = 10      # real world panda height to reset to before a grasp
scale_gauge_data = 1.0          # scale real gauge data by this
scale_wrist_data = 1.0          # scale real wrist data by this
scale_palm_data = 1.0           # scale real palm data by this
image_rate = 1                  # 1=take pictures every step, 2=every two steps etc
image_batch_size = 1            # 1=save pictures every trial, 2=every two trials etc

# important paths
test_save_path = "/home/luke/gripper-ros/"
mymujoco_rl_path = "/home/luke/mymujoco/rl"
pyfranka_path = "/home/luke/franka/franka_interface/build"

# experimental feature settings
use_sim_ftsensor = False        # use a simulated ft sensor instead of real life
dynamic_recal_ftsensor = False  # recalibrate ftsensor to zero whenever connection is lost
# sim_ft_sensor_step_offset = 2   # lower simulated ft sensor gripper down X steps
# prevent_back_palm = False       # swap any backward palm actions for forwards
prevent_x_open = False           # swap action 1 (X open) for action 2 (Y close)
render_sim_view = False         # render simulated gripper (CRASHES ON 2nd GRASP)
quit_on_palm = None               # quit grasping with a palm value above this (during test only), set None for off

# global flags
move_gripper = False            # is the gripper allowed to move
move_panda = False              # is the panda allowed to move
new_demand = False              # have we had a new demand for action
model_loaded = False            # has the dqn model been loaded
ready_for_new_action = False    # is the gripper ready for a new action
continue_grasping = True        # is grasping currently in progress and continuing
currently_testing = False       # is a test currently in progress

# declare global variables, these will be overwritten
step_num = 0                    # number of steps in our grasp
panda_z_height = 0.0            # panda z height state reading (=0 @ 10mm above gnd)
current_test_data = None        # current live test data structure
ftenv = None                    # simulated environment for fake force/torque sensor readings
norm_state_pub = None           # ROS publisher for normalised state readings
norm_sensor_pub = None          # ROS publisher for normalised sensor readings
debug_pub = None                # ROS publisher for gripper debug information
last_ft_value = None            # last ftsensor value, used to detect lost connection

# insert the mymujoco path for TrainDQN.py and ModelSaver.py files
sys.path.insert(0, mymujoco_rl_path)
from TrainDQN import TrainDQN
from ModelSaver import ModelSaver

# create model instance
device = "cuda" #none
dqn_log_level = 1 if log_level > 1 else 0
model = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)

# create modelsaver instance for saving test data
testsaver = ModelSaver("test_data", root=test_save_path)

# ----- callbacks and functions to run grasping ----- #

def data_callback(state):
  """
  Receives new state data for the gripper
  """

  global model

  # we cannot start saving data until the model class is ready (else silent crash)
  if not model_loaded: return

  # create vectors of observation data, order is very important!
  state_vec = [
    state.pose.x, 
    state.pose.y, 
    state.pose.z, 
    panda_z_height
  ]

  # optional: scale data
  global scale_gauge_data, scale_palm_data, scale_wrist_data
  state.sensor.gauge1 *= scale_gauge_data
  state.sensor.gauge2 *= scale_gauge_data
  state.sensor.gauge3 *= scale_gauge_data
  state.sensor.gauge4 *= scale_palm_data
  state.ftdata.force.z *= scale_wrist_data

  # if use a simulated ftsensor, override with simulated data
  if use_sim_ftsensor:
    state.ftdata.force.z = model.env.mj.sim_sensors.read_wrist_Z_sensor()

  # if we rezero ftsensor every time values repeat (ie connection lost)
  if dynamic_recal_ftsensor:
    global last_ft_value
    # rospy.loginfo(f"last value = {last_ft_value}, new value = {state.ftdata.force.z}")
    # if new value and old are exactly floating-point equal
    if state.ftdata.force.z == last_ft_value:
      model.env.mj.real_sensors.wrist_Z.offset = last_ft_value
      # if log_level > 1:
      #   rospy.loginfo("RECALIBRATED FTSENSOR TO ZERO")
    last_ft_value = state.ftdata.force.z

  # assemble our state vector (order is vitally important! Check mjclass.cpp)
  sensor_vec = [
    state.sensor.gauge1, 
    state.sensor.gauge2, 
    state.sensor.gauge3, 
    state.sensor.gauge4,
    state.ftdata.force.z
  ]

  # input the data
  model.env.mj.input_real_data(state_vec, sensor_vec)

  # publish the dqn network normalised input values
  global norm_state_pub, norm_sensor_pub
  norm_state = NormalisedState()
  norm_sensor = NormalisedSensor()

  # can visualise 'raw' data, 'SI' calibrated data, or 'normalised' network input data
  norm_state.gripper_x = model.env.mj.real_sensors.normalised.read_x_motor_position()
  norm_state.gripper_y = model.env.mj.real_sensors.normalised.read_y_motor_position()
  norm_state.gripper_z = model.env.mj.real_sensors.normalised.read_z_motor_position()
  norm_state.base_z = model.env.mj.real_sensors.normalised.read_z_base_position()

  norm_sensor.gauge1 = model.env.mj.real_sensors.normalised.read_finger1_gauge()
  norm_sensor.gauge2 = model.env.mj.real_sensors.normalised.read_finger2_gauge()
  norm_sensor.gauge3 = model.env.mj.real_sensors.normalised.read_finger3_gauge ()
  norm_sensor.palm = model.env.mj.real_sensors.normalised.read_palm_sensor()
  norm_sensor.wrist_z = model.env.mj.real_sensors.normalised.read_wrist_Z_sensor()

  # if using a simulated ft sensor, replace with this data instead
  if use_sim_ftsensor:
    norm_sensor.wrist_z = model.env.mj.sim_sensors.read_wrist_Z_sensor()

  norm_state_pub.publish(norm_state)
  norm_sensor_pub.publish(norm_sensor)

  global ready_for_new_action, action_received

  # once an action is completed, we need a new one
  if state.is_target_reached:

    global ready_for_new_action
    ready_for_new_action = True

def generate_action():
  """
  Get a new gripper state
  """

  global currently_testing
  global current_test_data

  obs = model.env.mj.get_real_observation()
  torch_obs = model.to_torch(obs)
  
  if currently_testing and current_test_data.data.heuristic:
    # if we are doing heuristic grasping
    action = model.env.get_heuristic_action()
  else:
    # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
    action = model.select_action(torch_obs, decay_num=1, test=True)
    action = action.item()

  if log_level > 1: rospy.loginfo(f"Generated action, action code is: {action}")

  # # if we are preventing palm backwards actions
  # if prevent_back_palm and action == 5:
  #   if log_level > 0: rospy.loginfo(f"prevent_back_palm=TRUE, backwards palm prevented")
  #   action = 4

  # prevent x open
  if prevent_x_open and action == 1:
    if log_level > 0: rospy.loginfo(f"prevent_x_open=TRUE, setting Y close instead")
    action = 2

  # apply the action and get the new target state (vector)
  new_target_state = model.env.mj.set_action(action)

  # if using a simulated ft sensor, resolve the action in simulation
  if use_sim_ftsensor or render_sim_view:
    if log_level > 0: rospy.loginfo("use_sim_ftsensor=TRUE, taking simulated action")
    model.env.mj.action_step()
    if render_sim_view: model.env.mj.render()

  # if at test time, save simple, unnormalised state data for this step
  if currently_testing:
    SI_state_vector = model.env.mj.get_simple_state_vector(model.env.mj.real_sensors.SI)
    current_test_data.add_step(obs, action, SI_vector=SI_state_vector)
    if quit_on_palm is not None:
      palm_force = model.env.mj.real_sensors.SI.read_palm_sensor()
      if palm_force > quit_on_palm:
        print(f"PALM FORCE OF {palm_force:.1f} UNSAFE, CANCELLING GRASPING")
        cancel_grasping_callback()

  # determine if this action is for the gripper or panda
  if model.env.mj.last_action_gripper(): for_franka = False
  elif model.env.mj.last_action_panda(): for_franka = True
  else: raise RuntimeError("last action not on gripper or on panda")

  return new_target_state, for_franka

def move_panda_z_abs(franka, target_z):
  """
  Move the panda to a new z position with cartesian motion using Valerio's library,
  an instance of which is passed as 'franka'
  """

  global panda_z_height

  # create identity matrix (must be floats!)
  T = np.array(
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]],
     dtype=np.float
  )

  # determine the change in the z height
  z_change = target_z - panda_z_height

  # insert z target into matrix
  T[2,3] = z_change

  rospy.loginfo(f"New panda target is {target_z * 1000:.1f} mm, old={panda_z_height * 1000:.1f} mm, change is {z_change * 1000:.1f} mm")

  # hardcoded safety checks
  min = -30e-3
  max = 30e-3
  if target_z < min:
    rospy.logwarn(f"panda z target of {target_z} is below the minimum of {min}")
    return
  if target_z > max:
    rospy.logwarn(f"panda z target of {target_z} is above the maximum of {max}")
    return

  # define duration in seconds (too low and we get acceleration errors)
  duration = 1.0

  franka.move("relative", T, duration)

  # update with the new target position
  panda_z_height = target_z

  return

def execute_grasping_callback(request=None):
  """
  Service callback to complete a dqn grasping task
  """

  if log_level > 0: rospy.loginfo("dqn node is now starting a grasping task")

  global ready_for_new_action
  global action_delay
  global continue_grasping
  global panda_z_height
  global step_num

  reset_all() # note this calls 'cancel_grasping_callback' if continue_grasping == True

  # this flag allows grasping to proceed
  continue_grasping = True

  step_num = 0

  rate = rospy.Rate(20)

  while not rospy.is_shutdown() and continue_grasping:

    if ready_for_new_action and model_loaded:

      ready_for_new_action = False

      # if we have reached our target position, terminate grasping
      if panda_z_height > 30e-3 - 1e-6:
        if log_level > 0: rospy.loginfo("Panda height has reached 30mm, stopping grasping")
        cancel_grasping_callback()

      # if we have reached our step limit, terminate grasping
      if step_num > model.env.params.max_episode_steps:
        if log_level > 0: rospy.loginfo(f"Max episode steps of {model.env.params.max_episode_steps} exceeded, stopping grasping")
        cancel_grasping_callback()

      # evaluate the network and get a new action
      step_num += 1
      if log_level > 0: rospy.loginfo(f"Grasping step {step_num}")
      new_target_state, for_franka = generate_action()

      # do we delay before performing action (eg 0.5 seconds)
      if action_delay is not None:
        if log_level > 1: 
          rospy.loginfo(f"Sleeping before action execution for {action_delay} seconds")
        rospy.sleep(action_delay)
        if log_level > 1: rospy.loginfo("Finished sleeping")

      # has the grasping task been cancelled
      if not continue_grasping:
        if log_level > 1: rospy.loginfo("Grasping cancelled, action not executed") 
        break

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
        if log_level > 1: rospy.loginfo(f"dqn is sending a panda control signal to z = {panda_target:.6f}")
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

  # ensure grasping is cancelled
  if continue_grasping:
    cancel_grasping_callback()

  # reset the gripper position (not blocking, publishes request)
  if move_gripper:
    if log_level > 1: rospy.loginfo("dqn node is about to try and reset gripper position")
    reset_gripper()
  elif log_level > 1: rospy.loginfo("dqn not reseting gripper position as move_gripper is False")

  # reset the panda position (blocking)
  if move_panda:
    if log_level > 1: rospy.loginfo(f"dqn node is about to try and reset panda position to {panda_reset_height_mm}")
    panda_reset_req = ResetPanda()
    panda_reset_req.reset_height_mm = panda_reset_height_mm # dqn episode begins at 10mm height
    reset_panda(panda_reset_req)
  elif log_level > 1: rospy.loginfo("dqn not reseting panda position as move_panda is False")

  # reset and prepare environment
  rospy.sleep(2.0)  # sleep to allow sensors to settle before recalibration
  model.env.mj.calibrate_real_sensors()
  model.env.reset() # recalibrates sensors
  model.env.mj.reset_object() # remove any object from the scene in simulation

  if currently_testing and current_test_data.data.heuristic:
    model.env.start_heuristic_grasping(realworld=True)

  # check for settings clashes
  if use_sim_ftsensor and dynamic_recal_ftsensor:
      if log_level > 0: 
        rospy.logwarn("use_sim_ftsensor and dynamic_recal_ft_sensor both True at the same time")

  if log_level > 1: rospy.loginfo("dqn node reset_all() is finished, sensors recalibrated")

  return []

def reset_panda(request=None):
  """
  Reset the panda position to a hardcoded joint state.
  """

  global franka_instance
  global move_panda
  global log_level
  global panda_z_height

  if not move_panda:
    if log_level > 0: rospy.logwarn("asked to reset_panda() but move_panda is false")
    return False

  if request is None:
    reset_to = 10 # default, panda is limited to +-30mm in move_panda_z_abs
  else:
    reset_to = request.reset_height_mm

  # new calibrations 22/3/23, 0=firm neoprene press, -6=fails
  cal_0_mm = [-0.04460928, 0.38211982, -0.00579623, -1.20211819, -0.01439458, 1.61249259, 0.75540974]
  cal_2_mm = [-0.04462827, 0.38808571, -0.00581715, -1.18840848, -0.01443909, 1.59992939, 0.75540858]
  cal_4_mm = [-0.04465780, 0.39017143, -0.00584690, -1.18030717, -0.01449511, 1.59377803, 0.75539456]
  cal_6_mm = [-0.04465780, 0.39229479, -0.00584523, -1.17205779, -0.01450923, 1.58758239, 0.75534843]
  cal_8_mm = [-0.04465585, 0.39438331, -0.00584356, -1.16354128, -0.01450720, 1.58123078, 0.75526212]
  cal_10_mm = [-0.04465586, 0.39663358, -0.00584523, -1.15490750, -0.01450791, 1.57477994, 0.75516820]
  cal_12_mm = [-0.04481524, 0.40074865, -0.00589660, -1.14791929, -0.01460787, 1.56819993, 0.75505090]
  cal_14_mm = [-0.04481524, 0.40295305, -0.00589637, -1.13912490, -0.01460929, 1.56178082, 0.75500080]
  cal_16_mm = [-0.04481162, 0.40537550, -0.00589997, -1.13007187, -0.01460378, 1.55529318, 0.75492834]
  cal_18_mm = [-0.04481337, 0.40787476, -0.00589885, -1.12109334, -0.01459970, 1.54875262, 0.75484305]
  cal_20_mm = [-0.04469420, 0.41042913, -0.00585969, -1.11176795, -0.01456532, 1.54207928, 0.75484156]
  cal_30_mm = [-0.04473575, 0.42504616, -0.00586498, -1.06289308, -0.01454851, 1.50777675, 0.75434994]
  cal_40_mm = [-0.04478521, 0.44279476, -0.00585969, -1.00850484, -0.01452637, 1.47115869, 0.75380754]
  cal_50_mm = [-0.04487317, 0.46457349, -0.00585969, -0.94686224, -0.01449903, 1.43148042, 0.75321650]

  calibrated_0mm = cal_2_mm       # what joints for the floor
  calibrated_start = cal_12_mm    # what start position before grasping, can adjust

  # find which hardcoded joint state to reset to
  reset_to = int(reset_to + 0.5)

  if reset_to == 0: target_state = calibrated_0mm
  elif reset_to == 10: target_state = calibrated_start
  elif reset_to == 20: target_state = cal_20_mm
  elif reset_to == 30: target_state = cal_30_mm
  elif reset_to == 40: target_state = cal_40_mm
  elif reset_to == 50: target_state = cal_50_mm
  else:
    raise RuntimeError(f"reset_panda given target reset of {reset_to} which does not correspond to known reset")

  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.1 # 0.1 is slow movements
  # franka_instance.move_joints(target_50_mm, speed_factor) # first move to safe height
  franka_instance.move_joints(target_state, speed_factor) # now approach reset height

  panda_z_height = 0

  if log_level > 0: rospy.loginfo(f"panda reset to a height of {reset_to}")

  return True

def reset_gripper(request=None):
  """
  Publish a homing command for the gripper
  """

  global demand_pub
  global move_gripper

  if not move_gripper:
    if log_level > 0: rospy.logwarn("asked to reset_gripper() but move_gripper is false")
    return

  # create a homing request for the gripper
  homing_demand = GripperDemand()
  homing_demand.home = True

  if log_level > 0: rospy.loginfo("dqn node requests homing gripper position")
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
    sys.path.insert(0, pyfranka_path)
    
    import pyfranka_interface

    # create franka controller instance
    franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)
    move_panda = True
    if log_level > 0: rospy.loginfo("Panda connection started successfully")

    return []

  except Exception as e:

    rospy.logerr(e)
    rospy.logerr("Failed to start panda conenction")

    # handle_panda_error()
    # franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)
    # move_panda = True
    # if log_level > 0: rospy.loginfo("Panda connection started successfully")

    move_panda = False

def handle_panda_error(request=None):
  """
  Return information and try to reset a panda error
  """

  # this function crashes the program, this idea did not work
  return

  global franka_instance

  if franka_instance.isError():

    # return information about the error
    print("Franka currently has an error:", franka_instance.getErrorString())

    # try to reset the error
    franka_instance.automaticErrorRecovery()

  return

def debug_gripper(request=None):
  """
  Request debug information from the gripper
  """

  global debug_pub

  if log_level > 1: rospy.loginfo("dqn node requesting gripper debug information")

  req = GripperInput()
  req.print_debug = True
  debug_pub.publish(req)

  return []

def load_model(request):
  """
  Load a dqn model
  """

  # PUT IN BEST_ID CODE? maybe not as only copy across one model
  
  global model_loaded
  global model, dqn_log_level

  model = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)

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

    # overwrite model 'run' and 'group' names with original
    model.run_name = request.run_name
    model.group_name = request.group_name

    # noise settings for real data
    # model.env.mj.set.state_noise_std = 0.0            # no state noise
    # model.env.mj.set.sensor_noise_std = 0.0           # no sensor noise
    # model.env.mj.set.state_noise_mu = 0.0             # no mean noise
    # model.env.mj.set.sensor_noise_mu = 0.0            # no mean noise

    model.env.mj.set.set_use_noise(False) # disable all noise
    model.env.reset()

    # IMPORTANT: have to run calibration function to setup real sensors
    model.env.mj.calibrate_real_sensors()

    # # if we are using a simulated ftsensor
    # global use_sim_ftsensor, ftenv
    # if use_sim_ftsensor:
    #   use_sim_ftsensor = False
    #   if log_level > 0: rospy.loginfo("USE_SIM_FTSENSOR=TRUE, creating ftenv now")
    #   tempmodel = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)
    #   tempmodel.load(id=request.run_id, folderpath=pathtofolder, foldername=foldername)
    #   ftenv = tempmodel.env #deepcopy(model.env)
    #   ftenv.mj.reset()
    #   if log_level > 0: rospy.loginfo("ftenv created")
    #   use_sim_ftsensor = True

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

  if request.default:
    # settings overriden with default values
    log_level = 2
    action_delay = 0.2
    move_gripper = True
    move_panda = True

  else:
    # settings overriden with user specified values
    log_level = request.log_level
    action_delay = request.action_delay
    move_gripper = request.move_gripper
    move_panda = request.move_panda

  # if starting or ending a panda connection
  if move_panda: connect_panda()
  else: franka_instance = None

  return []

def start_test(request):
  """
  Begin a full grasping test
  """

  if log_level > 0: rospy.loginfo(f"Starting a new test with name: {request.name}")

  if not depth_camera_connected:
    rospy.logwarn("Depth camera is not connected, test aborted")
    return False

  global testsaver, current_test_data, model, currently_testing, image_rate
  testsaver.enter_folder(request.name, forcecreate=True)
  current_test_data = GraspTestData() # wipe data clean
  current_test_data.start_test(request.name, model, get_depth_image, image_rate=image_rate)
  currently_testing = True

  # save the model in use if this file does not exist already
  savepath = testsaver.get_current_path()
  if not os.path.exists(savepath + "dqn_model" + testsaver.file_ext):
    testsaver.save("dqn_model", pyobj=model, suffix_numbering=False)
  else:
    if log_level > 1:
      rospy.loginfo("start_test() is not saving the dqn model as it already exists")

  # load any existing test data
  recent_data = testsaver.get_recent_file(name="test_data")
  if recent_data is not None:
    if log_level > 1:
      rospy.loginfo("start_test() found existing test data, loading it now")
    current_test_data.data = testsaver.load(fullfilepath=recent_data)

  if log_level > 0: rospy.loginfo("Ready to start testing")

  return []

def heuristic_test(request):
  """
  Begin a heuristic grasping test
  """

  start_test(request)
  make_test_heurisitc()

  return []

def end_test(request):
  """
  Finish a grasping test
  """

  global current_test_data, currently_testing

  if log_level > 0: 
    rospy.loginfo(f"Ending test with name: {current_test_data.data.test_name}")
  if log_level > 1: rospy.loginfo(f"Saving test data now...")
  
  testsaver.save("test_data", pyobj=current_test_data.data)

  if log_level > 1: rospy.loginfo("...finished saving test data")

  currently_testing = False

  return []

def start_trial(request):
  """
  Start a trial, testing grasping on a new object
  """

  if not currently_testing:
    rospy.logwarn("start_trial(...) called but not currently testing, first call start_test(...)")
    return False

  if log_level > 1:
    rospy.loginfo(f"Starting trial: Object={request.object_name} (num={request.object_number}), trial_num={request.trial_number}")

  global current_test_data
  current_test_data.start_trial(request.object_name, request.object_number, request.trial_number)

  # grasp
  execute_grasping_callback()

  return []

def save_trial(request):
  """
  Save trial data because the trial is finished
  """

  if not currently_testing:
    rospy.logwarn("save_trial(...) called but not currently testing, first call start_test(...)")
    return False

  # first, reset the scene as the trial is over
  # do this so I can place a new object whilst pickle is saving the image data
  try:
    reset_all()
  except Exception as e:
    rospy.logwarn(f"Unable to reset in save_trail(), error: {e}")

  global current_test_data
  current_test_data.finish_trial(request)

  global image_batch_size
  if image_batch_size:
    num_trials = len(current_test_data.image_data.trials)
    if num_trials >= image_batch_size:

      if log_level > 1: rospy.loginfo("Saving batch of image data...")

      # save
      testsaver.save("trial_image_batch", pyobj=current_test_data.image_data)

      # wipe old images
      current_test_data.image_data.trials = []

      # save temporary data of the test in case the program crashes
      testsaver.save("temp_test_data", pyobj=current_test_data.data,
                 suffix_numbering=False)

      if log_level > 1: rospy.loginfo("Saving batch of image data complete")

  if log_level > 1:
    rospy.loginfo("Trial data saved")

  return []

def delete_last_trial(request=None):
  """
  Delete the last saved trial, along with image data
  """

  global current_test_data

  # remove the information about the last trial
  try:
    current_test_data.data.trials.pop()
  except IndexError as e:
    rospy.logwarn(f"delete_last_trial() tried to pop empty data list: {e}")
  try:
    current_test_data.image_data.trials.pop()
  except IndexError as e:
    rospy.loginfo(f"delete_last_trial() tried to pop empty image_data list: {e}")

  # 'delete' any saved image data by renaming (temp test data will be overwritten)
  most_recent_file = testsaver.get_recent_file(name="trial_image_batch")

  if most_recent_file is None:
    rospy.logwarn("delete_last_trial() could not find any recent 'trial_image_batch' files")
    return []

  time_now = datetime.now().strftime("%d-%m-%y-%H:%M")
  os.rename(most_recent_file, most_recent_file + ".deleted_" + time_now)

  return []

def print_test_results(request=None):
  """
  Print test result details
  """

  global current_test_data
  details_str = current_test_data.get_test_string(print_trials=True)

  rospy.loginfo(details_str)

  return []

def load_baseline_1_model(request=None):
  """
  Load a new dqn model
  """

  # defaults
  folderpath = "/home/luke/mymujoco/rl/models/dqn/paper_baseline_1"
  id = None

  tol = 1e-5

  if (abs(request.thickness - 9e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 0
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 24e-3) < tol):
     finger = 1
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 2
  else:
    rospy.logwarn(f"Width={request.width} and thickness={request.thickness} not valid in load_baseline_model()")

  if finger == 0:

    rospy.loginfo("Loading model for finger 0.9mm thick and 28mm width")

    if request.sensors == 0:
      group = "16-01-23"
      name = "luke-PC_14_21_A10"

    elif request.sensors == 1:
      group = "16-01-23"
      name = "luke-PC_14_21_A39"

    elif request.sensors == 2:
      group = "16-01-23"
      name = "luke-PC_14_21_A72"

    elif request.sensors == 3:
      group = "23-01-23"
      name = "luke-PC_11_18_A102"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 1:

    rospy.loginfo("Loading model for finger 1.0mm thick and 24mm width")
    rospy.logwarn("1.0mm thick 24mm width finger not implemented yet")
    return

  elif finger == 2:

    rospy.loginfo("Loading model for finger 1.0mm thick and 28mm width")

    if request.sensors == 0:
      group = "16-01-23"
      name = "luke-PC_14_21_A25"

    elif request.sensors == 1:
      group = "16-01-23"
      name = "luke-PC_14_21_A59"

    elif request.sensors == 2:
      group = "23-01-23"
      name = "luke-PC_11_18_A84"

    elif request.sensors == 3:
      group = "23-01-23"
      name = "luke-PC_11_18_A116"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_new_model()")

  rospy.loginfo(f"Number of sensors is {request.sensors}")
  rospy.loginfo(f"Group name: {group}, run name: {name}")

  load = LoadModel()
  load.folderpath = folderpath
  load.group_name = group
  load.run_name = name
  load.run_id = id
  load_model(load)

  return True

def load_baseline_3_model(request=None):
  """
  Load a new dqn model
  """

  # defaults
  folderpath = "/home/luke/mymujoco/rl/models/dqn/paper_baseline_3"
  id = None

  tol = 1e-5

  if (abs(request.thickness - 9e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 0
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 24e-3) < tol):
     finger = 1
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 2
  else:
    rospy.logwarn(f"Width={request.width} and thickness={request.thickness} not valid in load_baseline_model()")

  if finger == 0:

    rospy.loginfo("Loading model for finger 0.9mm thick and 28mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      group = "03-03-23"
      name = "luke-PC_13_10_A70"

    elif request.sensors == 2:
      group = "03-03-23"
      name = "luke-PC_13_10_A140"

    elif request.sensors == 3:
      group = "24-02-23"
      name = "luke-PC_18_17_A37"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 1:

    rospy.loginfo("Loading model for finger 1.0mm thick and 24mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 2:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 3:
      group = "02-03-23"
      name = "luke-PC_16_12_A105"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 2:

    rospy.loginfo("Loading model for finger 1.0mm thick and 28mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 2:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 3:
      group = "27-02-23"
      name = "luke-PC_17_51_A118"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_new_model()")

  rospy.loginfo(f"Number of sensors is {request.sensors}")
  rospy.loginfo(f"Group name: {group}, run name: {name}")

  load = LoadModel()
  load.folderpath = folderpath
  load.group_name = group
  load.run_name = name
  load.run_id = id
  load_model(load)

  return True

def load_baseline_4_model(request=None):
  """
  Load a new dqn model
  """

  # defaults
  folderpath = "/home/luke/mymujoco/rl/models/paper_baseline_4"
  id = None

  tol = 1e-5

  if (abs(request.thickness - 9e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 0
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 24e-3) < tol):
     finger = 1
  elif (abs(request.thickness - 10e-4) < tol and
      abs(request.width - 28e-3) < tol):
     finger = 2
  else:
    rospy.logwarn(f"Width={request.width} and thickness={request.thickness} not valid in load_baseline_model()")

  if finger == 0:

    rospy.loginfo("Loading model for finger 0.9mm thick and 28mm width")

    if request.sensors == 0:
      group = "17-03-23"
      name = "luke-PC_13:15_A17"

    elif request.sensors == 1:
      group = "13-03-23"
      name = "luke-PC_17:23_A74"

    elif request.sensors == 2:
      group = "13-03-23"
      name = "luke-PC_17:23_A122"

    elif request.sensors == 3:
      group = "07-03-23"
      name = "luke-PC_13:37_A10"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 1:

    rospy.loginfo("Loading model for finger 1.0mm thick and 24mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      group = "31-03-23"
      name = "luke-PC_16:46_A90"

    elif request.sensors == 2:
      group = "27-03-23"
      name = "luke-PC_17:29_A142"

    elif request.sensors == 3:
      # group = "12-03-23"
      # name = "luke-PC_17:37_A220"
      # alternative training
      group = "06-04-23"
      name = "luke-PC_16:54_A217"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 2:

    rospy.loginfo("Loading model for finger 1.0mm thick and 28mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      group = "31-03-23"
      name = "luke-PC_16:46_A104"

    elif request.sensors == 2:
      group = "27-03-23"
      name = "luke-PC_17:29_A170"
      
    elif request.sensors == 3:
      group = "10-03-23"
      name = "luke-PC_17:27_A239"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_new_model()")

  rospy.loginfo(f"Number of sensors is {request.sensors}")
  rospy.loginfo(f"Group name: {group}, run name: {name}")

  load = LoadModel()
  load.folderpath = folderpath
  load.group_name = group
  load.run_name = name
  load.run_id = id
  load_model(load)

  return True

def set_sim_ft_sensor_callback(request):
  """
  Set the flag for using a simulated force torque sensor
  """

  global use_sim_ftsensor
  if log_level > 1: rospy.loginfo(f"use_sim_ftsensor originally set to {use_sim_ftsensor}")
  use_sim_ftsensor = request.data
  if log_level > 0: rospy.loginfo(f"use_sim_ftsensor is now set to {request.data}")

  return []

def set_dynamic_recal_ft_callback(request):
  """
  Set the bool value of whether we dynamically reset the ftsensor to zero
  every time we get two exactly repeated values (indicating a connection drop,
  which implies the sensor is in steady state and under no force)
  """

  global dynamic_recal_ftsensor
  if log_level > 1: rospy.loginfo(f"dynamic_recal_ftsensor originally set to {dynamic_recal_ftsensor}")
  dynamic_recal_ftsensor = request.data
  if log_level > 0: rospy.loginfo(f"dynamic_recal_ftsensor is now set to {dynamic_recal_ftsensor}")

  return []

def make_test_heurisitc(request=None):
  """
  Set the given test to be heuristic grasping
  """
  global current_test_data
  if log_level > 1: rospy.loginfo(f"test heuristic was set to {current_test_data.data.heuristic}")
  current_test_data.data.heuristic = True
  if log_level > 0: rospy.loginfo(f"test heurisitc now set to TRUE")

  return []

# ----- scripting to initialise and run node ----- #

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("dqn_node")
  rospy.loginfo("dqn node main has now started")
  
  # what namespace will we use in this node for publishers/services
  node_ns = "dqn" # gripper/dqn

  # get input parameters - are we allowing robot movement?
  move_gripper = True # rospy.get_param(f"/{node_ns}/move_gripper")
  move_panda = True # rospy.get_param(f"/{node_ns}/dqn/move_panda")
  rospy.loginfo(f"move gripper is {move_gripper}")
  rospy.loginfo(f"move panda is {move_panda}")

  # do we need to import the franka control library
  if move_panda: connect_panda()
    
  # subscriber for gripper data and publisher to send gripper commands
  rospy.Subscriber("/gripper/real/state", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher(f"/{node_ns}/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher(f"/{node_ns}/sensor", NormalisedSensor, queue_size=10)
  debug_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)

  # user set - what do we load by default
  if True:

    # load a model with a given path
    load = LoadModel()
    load.folderpath = "/home/luke/mymujoco/rl/models/dqn/"
    # load.folderpath += "paper_baseline_3_extra/"
    load.group_name = "26-04-23"
    load.run_name = "luke-PC_15:26_A64"
    load.run_id = None
    load_model(load)

  else:

    # load a specific model baseline
    load = LoadBaselineModel()
    load.thickness = 0.9e-3
    load.width = 28e-3
    load.sensors = 3
    load_baseline_4_model(load)

  # # uncomment for more debug information
  # model.env.log_level = 2
  # model.env.mj.set.debug = True

  # begin services for this node
  rospy.loginfo(f"dqn node services now available under namespace /{node_ns}/")
  rospy.Service(f"/{node_ns}/start", Empty, execute_grasping_callback)
  rospy.Service(f"/{node_ns}/stop", Empty, cancel_grasping_callback)
  rospy.Service(f"/{node_ns}/reset", Empty, reset_all)
  rospy.Service(f"/{node_ns}/reset_panda", ResetPanda, reset_panda)
  rospy.Service(f"/{node_ns}/reset_gripper", Empty, reset_gripper)
  rospy.Service(f"/{node_ns}/connect_panda", Empty, connect_panda)
  rospy.Service(f"/{node_ns}/load_model", LoadModel, load_model)
  rospy.Service(f"/{node_ns}/apply_settings", ApplySettings, apply_settings)
  rospy.Service(f"/{node_ns}/debug_gripper", Empty, debug_gripper)
  rospy.Service(f"/{node_ns}/test", StartTest, start_test)
  rospy.Service(f"/{node_ns}/heuristic_test", StartTest, heuristic_test)
  rospy.Service(f"/{node_ns}/trial", StartTrial, start_trial)
  rospy.Service(f"/{node_ns}/end_test", Empty, end_test)
  rospy.Service(f"/{node_ns}/save_trial", SaveTrial, save_trial)
  rospy.Service(f"/{node_ns}/delete_trial", Empty, delete_last_trial)
  rospy.Service(f"/{node_ns}/print_test", Empty, print_test_results)
  rospy.Service(f"/{node_ns}/load_baseline_model", LoadBaselineModel, load_baseline_4_model)
  rospy.Service(f"/{node_ns}/set_use_sim_ft_sensor", SetBool, set_sim_ft_sensor_callback)
  rospy.Service(f"/{node_ns}/set_dynamic_recal_ft", SetBool, set_dynamic_recal_ft_callback)
  rospy.Service(f"/{node_ns}/make_test_heuristic", Empty, make_test_heurisitc)

  try:
    while not rospy.is_shutdown(): rospy.spin() # and wait for service requests
  except Exception as e:
    end_test()
    rospy.logerror(f"dqn_node(...) failed, saved test data - exception is {e}")
  