#!/usr/bin/env python3

import rospy
import sys
import numpy as np

from std_srvs.srv import Empty
from gripper_msgs.msg import GripperState, GripperDemand, GripperInput
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings, ResetPanda

# insert the mymujoco path for TrainDQN.py file
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
device = "cuda" #none
dqn_log_level = 1
model = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)

# user settings defaults for this node
log_level = 2
action_delay = 0.0
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
debug_pub = None

# start position should always be zero (fingers should be 10mm above the ground)
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

  # TESTING scale up gauge data
  state.sensor.gauge1 *= 1.5
  state.sensor.gauge2 *= 1.5
  state.sensor.gauge3 *= 1.5

  sensor_vec = [
    state.sensor.gauge1, state.sensor.gauge2, state.sensor.gauge3, 
    state.sensor.gauge4,
    state.ftdata.force.z
  ]

  # input the data
  model.env.mj.input_real_data(state_vec, sensor_vec)

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

  # define duration in seconds
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

  # this flag allows grasping to proceed
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

  # reset the gripper position (not blocking, publishes request)
  if move_gripper:
    if log_level > 1: rospy.loginfo("dqn node is about to try and reset gripper position")
    reset_gripper()
  elif log_level > 1: rospy.loginfo("dqn not reseting gripper position as move_gripper is False")

  # reset the panda position (blocking)
  if move_panda:
    if log_level > 1: rospy.loginfo("dqn node is about to try and reset panda position")
    panda_reset_req = ResetPanda()
    panda_reset_req.reset_height_mm = 10 # dqn episode begins at 10mm height
    reset_panda(panda_reset_req)
  elif log_level > 1: rospy.loginfo("dqn not reseting panda position as move_panda is False")

  # reset and prepare environment
  rospy.sleep(2.0)  # sleep to allow sensors to settle before recalibration
  model.env.reset() # recalibrates sensors

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

  """
  Heights calibrated by hand, all fingers put to touching table (0mm), then use +mm motions
  Fingers touch table: 0mm
  Fingers first touch neoprene: 10mm
  Fingers all touch neprene: 8mm
  Lowest height possible before controller aborts (with neoprene): 2mm

  Hence: with neoprene, 0 position is 10mm
  """
  target_0_mm = [-0.05509114, 0.08494865, 0.00358691, -1.67752536, -0.00399310, 1.69452373, 0.84098394]
  target_6_mm = [-0.05525132, 0.08799861, 0.00338449, -1.66332317, -0.00395840, 1.67866505, 0.84087103]
  target_10_mm = [-0.05515629, 0.08709929, 0.00346314, -1.65161046, -0.00396003, 1.66868245, 0.84094963]
  target_16_mm = [-0.05524815, 0.08916576, 0.00338449, -1.63632929, -0.00395717, 1.65315411, 0.84086777]
  target_20_mm = [-0.05515193, 0.08893368, 0.00346119, -1.62365382, -0.00396328, 1.64260790, 0.84088849]
  target_26_mm = [-0.05524827, 0.09149043, 0.00338449, -1.60776543, -0.00395930, 1.62693296, 0.84084929]
  target_30_mm = [-0.05514753, 0.09176625, 0.00345952, -1.59441761, -0.00396328, 1.61631546, 0.84082899]
  target_36_mm = [-0.05523857, 0.09492149, 0.00338282, -1.57776986, -0.00396232, 1.60036405, 0.84081025]
  target_40_mm = [-0.05513947, 0.09566288, 0.00346119, -1.56383533, -0.00396267, 1.58957835, 0.84075323]
  target_46_mm = [-0.05523381, 0.09913632, 0.00338449, -1.54659639, -0.00396399, 1.57351597, 0.84075759]
  target_50_mm = [-0.05513489, 0.10048285, 0.00346119, -1.53197880, -0.00396267, 1.56242129, 0.84068724]
  target_56_mm = [-0.05521145, 0.10445451, 0.00338282, -1.51400512, -0.00396877, 1.54619678, 0.84069480]
  
  # find which hardcoded joint state to reset to (0 position = 10mm)
  reset_to = int(reset_to + 0.5) + 10

  if reset_to == 0: target_state = target_0_mm
  elif reset_to == 6: target_state = target_6_mm
  elif reset_to == 10: target_state = target_10_mm
  elif reset_to == 20: target_state = target_20_mm
  elif reset_to == 30: target_state = target_30_mm
  elif reset_to == 40: target_state = target_40_mm
  elif reset_to == 50: target_state = target_50_mm
  else:
    raise RuntimeError(f"reset_panda given target reset of {reset_to} which does not correspond to known reset")

  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.1 # 0.1 is slow movements
  franka_instance.move_joints(target_state, speed_factor)

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
    sys.path.insert(0, "/home/luke/franka_interface/build")
    import pyfranka_interface

    # create franka controller instance
    franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)
    move_panda = True
    if log_level > 0: rospy.loginfo("Panda connection started successfully")

    return []

  except Exception as e:

    rospy.logerr(e)
    rospy.logerr("Failed to start panda conenction")

    handle_panda_error()
    franka_instance = pyfranka_interface.Robot_("172.16.0.2", False, False)
    move_panda = True
    if log_level > 0: rospy.loginfo("Panda connection started successfully")

    move_panda = False

def handle_panda_error(request=None):
  """
  Return information and try to reset a panda error
  """

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

"""
Problems to address in this code:

1. Not repeatable, you can't run multiple grasps in a row as the gripper
does not work the 2nd time, it must not be being reset properly

2. panda z height, the controller can send signals that caused the panda to
hit the table and shut down, the code is unaware of this so grasping breaks

3. local minimum, in rare cases an action is continously chosen that does
nothing, like action 5 lifting the palm. It could be worth adding some noise
to sensor inputs to prevent this

4. ease of use, some functions are not that easy to use, for example stopping
involves cancelling the 'start' call and then writing 'stop'. One better system
could be to press the soft E-stop, if this code could detect when the franka
arm refuses to move. Then, this code should make it easier to reset everything
and start again

5. wrist z sensor signal resetting - this may be working but check again

6. ROS qt freezing - at first signals are seen live, but later in a grasp they
freeze frame, possibly when panda is moving or as network is evaluated

"""



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
  debug_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)

  # define the model to load and then try to load it
  load = LoadModel()
  load.folderpath = "/home/luke/mymujoco/rl/models/dqn/"
  # load.group_name = "02-12-22"
  # load.run_name = "luke-PC_16:55_A3"
  load.group_name = "06-01-23"
  load.run_name = "luke-PC_14:44_A48"
  load.run_id = None
  load_model(load)

  # noise settings for real data
  model.env.mj.set.state_noise_std = 0.025 # add noise to state readings
  model.env.mj.set.sensor_noise_std = 0.0  # do not add noise to sensor readings
  model.env.reset()

  # uncomment for more debug information
  # model.env.log_level = 2
  # model.env.mj.set.debug = True

  # begin services for this node
  rospy.loginfo("dqn node services now available")
  rospy.Service('/gripper/dqn/start', Empty, execute_grasping_callback)
  rospy.Service('/gripper/dqn/stop', Empty, cancel_grasping_callback)
  rospy.Service('/gripper/dqn/reset', Empty, reset_all)
  rospy.Service('/gripper/dqn/reset_panda', ResetPanda, reset_panda)
  rospy.Service("/gripper/dqn/reset_gripper", Empty, reset_gripper)
  rospy.Service("/gripper/dqn/connect_panda", Empty, connect_panda)
  rospy.Service("/gripper/dqn/load_model", LoadModel, load_model)
  rospy.Service("/gripper/dqn/apply_settings", ApplySettings, apply_settings)
  rospy.Service("/gripper/dqn/debug_gripper", Empty, debug_gripper)

  while not rospy.is_shutdown(): rospy.spin() # and wait for service requests
  