#!/usr/bin/env python3

import rospy
import sys
import os
import numpy as np
from datetime import datetime

from std_srvs.srv import Empty
from gripper_msgs.msg import GripperState, GripperDemand, GripperInput
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings, ResetPanda
from gripper_dqn.srv import StartTest, StartTrial, SaveTrial

from capture_depth_image import get_depth_image
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

# insert the mymujoco path for TrainDQN.py file
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
device = "cuda" #none
dqn_log_level = 1
model = TrainDQN(use_wandb=False, no_plot=True, log_level=dqn_log_level, device=device)

# create modelsaver instance
from ModelSaver import ModelSaver
saver = ModelSaver("test_data", root="/home/luke/gripper-ros/")

# user settings defaults for this node
log_level = 2
action_delay = 0.0
move_gripper = False
move_panda = False
scale_gauge_data = 1.0 # (was 1.5) scale real gauge data by this
scale_wrist_data = 1.0 # scale real wrist data by this
scale_palm_data = 1.0 # scale real palm data by this

# global flags
new_demand = False
model_loaded = False
ready_for_new_action = False
continue_grasping = True
currently_testing = False

# these will be ROS publishers once the node starts up
norm_state_pub = None
norm_sensor_pub = None
debug_pub = None

# start position should always be zero (fingers should be 10mm above the ground)
panda_z_height = 0.0

class GraspTestData:

  # data structures for saving testing data
  step_data = namedtuple("step_data", ("step_num", "state_vector", "action"))
  image_data = namedtuple("image_data", ("step_num", "rgb", "depth"))

  @dataclass
  class TrialData:
    object_name: str
    object_num: int
    trial_num: int
    steps: list
    images: list
    success: bool
    info: str

  @dataclass
  class TestData:
    trials: list
    test_name: str
    finger_width: float
    finger_thickness: float
    heuristic: bool
    bend_gauge: bool
    palm_sensor: bool
    wrist_Z_sensor: bool
    group_name: str
    run_name: str
    best_SR: float
    best_EP: float

  @dataclass
  class TestResults:
    num_trials: int
    num_objects: int
    avg_obj_num_trials: float
    success_rate: float
    avg_obj_success_rate: float
    sphere_SR: float
    cylinder_SR: float
    cuboid_SR: float
    cube_SR: float

  def __init__(self, test_name, dqn_obj, image_rate=1, heuristic=False):
    """
    Data for an entire test
    """

    best_sr, best_ep = dqn_obj.track.calc_best_performance()

    # create test data structure
    self.data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      dqn_obj.env.params.finger_width,          # finger width
      dqn_obj.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      dqn_obj.env.mj.set.bending_gauge.in_use,  # bending sensor
      dqn_obj.env.mj.set.palm_sensor.in_use,    # palm sensor
      dqn_obj.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      dqn_obj.group_name,                       # run group name
      dqn_obj.run_name,                         # run name
      best_sr,                                  # best test success rate in sim
      best_ep,                                  # episode at best_sr
    )

    self.image_data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      dqn_obj.env.params.finger_width,          # finger width
      dqn_obj.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      dqn_obj.env.mj.set.bending_gauge.in_use,  # bending sensor
      dqn_obj.env.mj.set.palm_sensor.in_use,    # palm sensor
      dqn_obj.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      dqn_obj.group_name,                       # run group name
      dqn_obj.run_name,                         # run name
      best_sr,                                  # best test success rate in sim
      best_ep,                                  # episode at best_sr
    )
    
    # initialise class variables
    self.image_rate = image_rate
    self.current_trial = None
    self.current_trial_with_images = None

  def start_trial(self, object_name, object_num, trial_num):
    """
    Begin a new trial
    """

    self.current_trial = GraspTestData.TrialData(
      object_name,    # object_name
      object_num,     # object_num
      trial_num,      # trial_num
      [],             # steps
      [],             # images
      None,           # success
      ""              # info
    )
    self.current_trial_with_images = GraspTestData.TrialData(
      object_name,    # object_name
      object_num,     # object_num
      trial_num,      # trial_num
      [],             # steps
      [],             # images
      None,           # success
      ""              # info
    )

    self.current_step_count = 0

  def add_step(self, state_vector, action):
    """
    Add data for a single step
    """

    if self.current_trial == None:
      raise RuntimeError("current_trial is None in GraspTestData class")

    self.current_step_count += 1

    # add step data to the current trial
    this_step = GraspTestData.step_data(
      self.current_step_count,    # step_num
      state_vector,               # state_vector
      action                      # action
    )
    self.current_trial.steps.append(this_step)

    # are we taking a photo this step
    if (self.current_step_count - 1) % self.image_rate == 0:
      rgb, depth = get_depth_image()
      this_image = GraspTestData.image_data(
        self.current_step_count,  # step_num
        rgb,                      # rgb
        depth                     # depth
      )
      # add step image pairs
      self.current_trial_with_images.steps.append(this_step)
      self.current_trial_with_images.images.append(this_image)

  def finish_trial(self, grasp_success, info):
    """
    Finish a trial and save the results
    """

    self.current_trial.success = grasp_success
    self.current_trial.info = info
    self.data.trials.append(deepcopy(self.current_trial))
    self.current_trial = None

    self.current_trial_with_images.success = grasp_success
    self.current_trial_with_images.info = info
    self.image_data.trials.append(deepcopy(self.current_trial_with_images))
    self.current_trial_with_images = None

  def get_test_results(self):
    """
    Get data structure of test information
    """
    entries = []
    object_nums = []
    entry = ["obj_name", "object_num", "num_trials", "num_successes", "info_strings"]
    entry[0] = ""
    entry[1] = 0
    entry[2] = 0
    entry[3] = 0
    entry[4] = []

    if len(self.data.trials) == 0:
      rospy.logwarn("get_test_results() found 0 trials, aborting")
      return None

    # sort trial data
    for trial in self.data.trials:

      found = False
      for j in range(len(object_nums)):
        if object_nums[j] == trial.object_num:
          found = True
          break

      if not found:

        # create a new entry
        new_entry = deepcopy(entry)
        new_entry[0] = trial.object_name
        new_entry[1] = trial.object_num
        new_entry[2] += 1
        new_entry[3] += trial.success
        new_entry[4].append(trial.info)

        entries.append(new_entry)
        object_nums.append(trial.object_num)

      else:

        # add to the existing entry
        entries[j][2] += 1
        entries[j][3] += trial.success
        entries[j].append(trial.info)

    # now process trial data
    object_SRs = []
    object_trials = []
    total_successes = 0
    for i in range(len(entries)):

      total_successes += entries[i][3]
      this_SR = (entries[i][3] / float(entries[i][2]))
      object_SRs.append(this_SR)
      object_trials.append(entries[i][2])
  
    # round up
    total_SR = total_successes / float(len(self.data.trials))
    avg_obj_SR = np.mean(np.array(object_SRs))
    avg_obj_trials = np.mean(np.array(object_trials))

    return GraspTestData.TestResults(
      len(self.data.trials),        # num_trials
      len(object_nums),             # num_objects
      avg_obj_trials,               # avg_obj_num_trials
      total_SR,                     # success_rate
      avg_obj_SR,                   # avg_obj_success_rate
      0.0,                          # sphere_SR
      0.0,                          # cylinder_SR
      0.0,                          # cuboid_SR
      0.0,                          # cube_SR
    )

  def get_test_string(self):
    """
    Print out information about the current test
    """

    info_str = """"""

    info_str += f"\nTest information\n\n"
    info_str += f"Test name: {current_test_data.data.test_name}\n"
    info_str += f"Finger width: {current_test_data.data.finger_width}\n"
    info_str += f"Finger thickness: {current_test_data.data.finger_thickness:.4f}\n"
    info_str += f"heuristic test: {current_test_data.data.heuristic}\n"
    info_str += f"Bending gauge in use: {current_test_data.data.bend_gauge}\n"
    info_str += f"Palm sensor in use: {current_test_data.data.palm_sensor}\n"
    info_str += f"Wrist Z sensor in use: {current_test_data.data.wrist_Z_sensor}\n"
    info_str += f"Loaded group name: {current_test_data.data.group_name}\n"
    info_str += f"Loaded run name: {current_test_data.data.run_name}\n"
    info_str += f"Loaded best SR: {current_test_data.data.best_SR}\n"

    results = self.get_test_results()

    if results is None: return

    info_str += f"\nResults information:\n\n"
    info_str += f"Total number of trials: {results.num_trials}\n"
    info_str += f"Total number of objects: {results.num_objects}\n"
    info_str += f"Avg. trials per object: {results.avg_obj_num_trials:.4f}\n"
    info_str += f"Overall success rate: {results.success_rate:.4f}\n"
    info_str += f"Avg. success rate per object: {results.avg_obj_success_rate:.4f}\n"
    info_str += f"Sphere success rate: {results.sphere_SR:.4f}\n"
    info_str += f"cylinder success rate: {results.cylinder_SR:.4f}\n"
    info_str += f"cuboid success rate: {results.cuboid_SR:.4f}\n"
    info_str += f"cube success rate: {results.cube_SR:.4f}\n"

    return info_str

# global for test data structure (GraspTestData)
current_test_data = None

"""
Each image is about 400kB (this is both rgb and depth) after compressed pickling.

That means that if each trial is 100 steps, it will be 40MB per trial.

If a test is made up of 30*5 trials, it will be 6GB per test.

This doesn't seem entirely unreasonable. Note that saving 6GB of data will take a LONG
time, could be several hours. Does the zotac have enough RAM for that? Considering the
uncompressed images will take up more space. I could have a folder for each test but then
save multiple times during the test.

Zotac has 16GB of RAM.

These things could be automated, so after we have done 30 trials it saves.

60.8MB took 36.8 seconds to save. Lets call it 0.5s per MB. That means 20seconds per trial
so a batch of 5 would take 1min 40s. The entire test of 150 trials would take 50mins.

Since I need to reset the object after each trial which takes some time, it is tempting to
save after every trial and use the time more efficiently. The downside is that we end up
with a lot of saved files (150 per test!)
"""
image_rate = 1

# are we autosaving images in batches of trials, 0 disables
image_batch_size = 1

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

  # optional: scale data
  global scale_gauge_data, scale_palm_data, scale_wrist_data
  state.sensor.gauge1 *= scale_gauge_data
  state.sensor.gauge2 *= scale_gauge_data
  state.sensor.gauge3 *= scale_gauge_data
  state.sensor.gauge4 *= scale_palm_data
  state.ftdata.force.z *= scale_palm_data

  # assemble our state vector (order is vitally important! Check mjclass.cpp)
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

  global currently_testing

  if log_level > 1: rospy.loginfo("generating a new action")

  obs = model.env.mj.get_real_observation()
  torch_obs = model.to_torch(obs)
  
  # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
  action = model.select_action(torch_obs, decay_num=1, test=True)

  if log_level > 1: rospy.loginfo(f"Action code is: {action.item()}")

  # apply the action and get the new target state (vector)
  new_target_state = model.env.mj.set_action(action.item())

  # if at test time, save data for this step
  if currently_testing:
    global current_test_data
    current_test_data.add_step(obs, action.item())

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
    cancel_grasping_callback() # stop any grasping once we hit maximum height
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
  model.env.mj.calibrate_real_sensors()
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

  # new calibrations 7/2/23 (0=firm touching neoprene, -6mm=possible, -8mm=failure)
  target_0_mm = [-0.05360574, 0.36224426, -0.02498490, -1.28254861, 0.00438162, 1.61758563, 0.82279294]
  target_10_mm = [-0.05384081, 0.37038012, -0.02496675, -1.24878968, 0.00448761, 1.58849793, 0.82276331]
  target_20_mm = [-0.05406938, 0.37825387, -0.02496313, -1.21092717, 0.00463154, 1.55859002, 0.82265671]
  target_30_mm = [-0.05441660, 0.38795699, -0.02496313, -1.17022414, 0.00483431, 1.52763658, 0.82253542]
  target_40_mm = [-0.05480234, 0.39970210, -0.02496146, -1.12615559, 0.00508573, 1.49539334, 0.82241654]
  target_50_mm = [-0.05526356, 0.41379487, -0.02496675, -1.07801807, 0.00540019, 1.46142950, 0.82228165]
 
  # find which hardcoded joint state to reset to
  reset_to = int(reset_to + 0.5)

  if reset_to == 0: target_state = target_0_mm
  elif reset_to == 10: target_state = target_10_mm
  elif reset_to == 20: target_state = target_20_mm
  elif reset_to == 30: target_state = target_30_mm
  elif reset_to == 40: target_state = target_40_mm
  elif reset_to == 50: target_state = target_50_mm
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

    # overwrite model 'run' and 'group' names with original
    model.run_name = request.run_name
    model.group_name = request.group_name

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

  global saver, current_test_data, model, currently_testing, image_rate
  saver.enter_folder(request.name, forcecreate=True)
  current_test_data = GraspTestData(request.name, model,
                                    image_rate=image_rate)
  currently_testing = True

  # save the model in use if this file does not exist already
  savepath = saver.get_current_path()
  if not os.path.exists(savepath + "dqn_model" + saver.file_ext):
    saver.save("dqn_model", pyobj=model, suffix_numbering=False)
  else:
    rospy.loginfo("start_test() is not saving the dqn model as it already exists")

  # load any existing test data
  recent_data = saver.get_recent_file(name="test_data")
  if recent_data is not None:
    rospy.loginfo("start_test() found existing test data, loading it now")
    current_test_data.data = saver.load(fullfilepath=recent_data)

  return []

def end_test(request):
  """
  Finish a grasping test
  """

  global current_test_data, currently_testing

  if log_level > 0: 
    rospy.loginfo(f"Ending test with name: {current_test_data.data.test_name}")
  if log_level > 1: rospy.loginfo(f"Saving test data now...")
  
  saver.save("test_data", pyobj=current_test_data.data)

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
  reset_all()

  global current_test_data
  current_test_data.finish_trial(request.grasp_success, request.info)

  global image_batch_size
  if image_batch_size:
    num_trials = len(current_test_data.image_data.trials)
    if num_trials >= image_batch_size:

      if log_level > 1: rospy.loginfo("Saving batch of image data...")

      # save
      saver.save("trial_image_batch", pyobj=current_test_data.image_data)

      # wipe old images
      current_test_data.image_data.trials = []

      # save temporary data of the test in case the program crashes
      saver.save("temp_test_data", pyobj=current_test_data.data,
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
  most_recent_file = saver.get_recent_file(name="trial_image_batch")

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
  details_str = current_test_data.get_test_string()

  rospy.loginfo(details_str)

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

  # IMPORTANT: have to run calibration function to setup real sensors
  model.env.mj.calibrate_real_sensors()

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
  rospy.Service("/gripper/dqn/test", StartTest, start_test)
  rospy.Service("/gripper/dqn/trial", StartTrial, start_trial)
  rospy.Service("/gripper/dqn/end_test", Empty, end_test)
  rospy.Service("/gripper/dqn/save_trial", SaveTrial, save_trial)
  rospy.Service("/gripper/dqn/delete_trial", Empty, delete_last_trial)
  rospy.Service("/gripper/dqn/print_test", Empty, print_test_results)

  try:
    while not rospy.is_shutdown(): rospy.spin() # and wait for service requests
  except Exception as e:
    end_test()
    rospy.logerror(f"dqn_node(...) failed, saved test data - exception is {e}")
  