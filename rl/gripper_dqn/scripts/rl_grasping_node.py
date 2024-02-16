#!/home/luke/pyenv/py38_ros/bin/python

# general imports
import sys
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp
import traceback
import time
import random

# ros specific imports
import rospy
from std_srvs.srv import Empty, SetBool
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Wrench
from gripper_msgs.msg import GripperState, GripperDemand, GripperInput
from gripper_msgs.msg import GripperOutput, GripperRequest
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings, ResetPanda
from gripper_dqn.srv import StartTest, StartTrial, SaveTrial, LoadBaselineModel
from gripper_dqn.srv import PandaMoveToInt, Demo, PandaMoveZ, ForceTest
from cv_bridge import CvBridge

# import the test data structure class
from grasp_test_data import GraspTestData

# ----- essential user settings ----- #

# important user settings
camera = True                  # do we want to take camera images
use_devel = True               # do we load trainings from mujoco-devel and run with that compilation
photoshoot_calibration = False  # do we grasp in ideal position for taking side-on-videos
use_panda_threads = True       # do we run panda in a seperate thread
use_panda_ft_sensing = True     # do we use panda wrench estimation for forces, else our f/t sensor
log_level = 2                   # node log level, 0=disabled, 1=essential, 2=debug, 3=all
action_delay = 0.1              # safety delay between action generation and publishing
panda_z_move_duration = 0.2    # how long in seconds to move the panda z height vertically up
panda_reset_height_mm = 10      # real world panda height to reset to before a grasp
panda_reset_noise_mm = 6        # noise on panda reset height, +- this value, multiples of 2 only
panda_target_height_mm = 20     # how high before a grasp is considered complete
scale_actions_on_load = 1.0     # scale action sizes upon loading a model
scale_gauge_data = 1.0          # scale real gauge data by this
scale_wrist_data = 1.0          # scale real wrist data by this
scale_palm_data = 1.0           # scale real palm data by this
image_rate = 1                  # 1=take pictures every step, 2=every two steps etc
image_batch_size = 1            # 1=save pictures every trial, 2=every two trials etc
use_forces_stable_grasp = True  # use force limits and gripper height to detect stable grasp
use_palm_force_test = True     # test grasp stability with palm disturbance
extra_gripper_measuring = True  # use a second gripper for measurements
palm_force_target = 10          # maximum force at which palm force test ends
XY_force_target = 10            # maximum force at which XY force test ends

# important paths
test_save_path = "/home/luke/gripper-ros/"
if use_devel: mymujoco_rl_path = "/home/luke/mujoco-devel/rl"
else: mymujoco_rl_path = "/home/luke/mymujoco/rl"
pyfranka_path = "/home/luke/libs/franka/franka_interface/build"

# experimental feature settings
debug_observation = False        # print debug information about the observation
print_network_eval_time = True  # print the time taken to evaluate the network
number_gripper_demands = 1      # how many gripper position commands to send to try ensure motions are achieved
use_sim_ftsensor = False        # use a simulated ft sensor instead of real life
dynamic_recal_ftsensor = False  # recalibrate ftsensor to zero whenever connection is lost
# sim_ft_sensor_step_offset = 2   # lower simulated ft sensor gripper down X steps
# prevent_back_palm = False       # swap any backward palm actions for forwards
prevent_x_open = False           # swap action 1 (X open) for action 2 (Y close)
render_sim_view = False         # render simulated gripper (CRASHES ON 2nd GRASP)
quit_on_palm = None               # quit grasping with a palm value above this (during test only), set None for off
reject_wrist_noise = False       # try to ignore large spikes in the wrist sensor
prevent_table_hit = False         # prevent the gripper from going below a certain height

# ----- global variable declarations ----- #

# global flags
move_gripper = False            # is the gripper allowed to move
move_panda = False              # is the panda allowed to move
new_demand = False              # have we had a new demand for action
model_loaded = False            # has the dqn model been loaded
ready_for_new_action = False    # is the gripper ready for a new action
continue_grasping = True        # is grasping currently in progress and continuing
currently_testing = False       # is a test currently in progress
continue_demo = False           # is the demo currently in progress and continuing
demo_panda_is_at = -1           # flag for the position the panda is in (only applies to a demo)
demo_loop_int = -1              # where in the demo loop are wek
risk_table_hit = False

# declare global variables, these will be overwritten
step_num = 0                    # number of steps in our grasp
franka_instance = None          # franka FCI connection class
panda_z_height = 0.0            # panda z height state reading (=0 @ 10mm above gnd)
current_test_data = None        # current live test data structure
ftenv = None                    # simulated environment for fake force/torque sensor readings
norm_state_pub = None           # ROS publisher for normalised state readings
norm_sensor_pub = None          # ROS publisher for normalised sensor readings
debug_pub = None                # ROS publisher for gripper debug information
panda_local_wrench_pub = None   # ROS publisher for panda external wrench estimates in local frame
panda_global_wrench_pub = None  # ROS publisher for panda external wrench estimates in global frame
last_ft_value = None            # last ftsensor value, used to detect lost connection
depth_camera_connected = False  # have we received images from the depth camera
rgb_image = None                # most recent rgb image
depth_image = None              # most recent depth image
image_observation = None        # most recent image observation (converted rgb -> torch tensor)
panda_in_motion = False         # is the panda currently moving
gripper_target_reached = False  # has the gripper reached its target
grasp_frc_stable = False        # is the gripper the appropriate height with forces stable
stable_grasp_frc = None         # measured forces in stable grasp
palm_SI_pub = None              # publisher for palm force in SI units (newtons)
extra_gripper_palm_frc = None   # palm force in newtons from the 2nd gripper (if in use)
extra_gripper_palm_raw = None   # palm force in newtons without any offset
extra_gripper_palm_pos = None   # palm position in metres from the 2nd gripper
extra_demand_pub = None         # demand publisher for the 2nd gripper
extra_gripper_z_offset = 0.0    # palm force offset in newtons, zeroed upon any reset if using 2nd gripper
extra_zfrc_pub = None           # palm force publisher for the 2nd gripper
trial_object_num = None         # number of the current object on trial
trial_object_axis = None        # name of the force axis to measure with the current object
axis_force_tol = None           # last force tolerated in a particular axis
prevent_force_test = False      # cancel signal to stop a force test from proceeding

# insert the mymujoco path for TrainDQN.py and ModelSaver.py files
sys.path.insert(0, mymujoco_rl_path)
from TrainingManager import TrainingManager
from ModelSaver import ModelSaver

# create model instance
device = "cpu" #none
dqn_log_level = 1 if log_level > 1 else 0
model = TrainingManager(log_level=dqn_log_level, device=device)

# create openCV bridge
bridge = CvBridge()

# create modelsaver instance for saving test data
testsaver = ModelSaver("test_data", root=test_save_path)

# ----- callbacks and functions to run grasping ----- #

# from: https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
class Process(mp.Process):
  def __init__(self, *args, **kwargs):
    mp.Process.__init__(self, *args, **kwargs)
    self._pconn, self._cconn = mp.Pipe()
    self._exception = None

  def run(self):
    max_tries = 1
    tries = 0
    while tries < max_tries:
      print("try", tries)
      try:
        mp.Process.run(self)
        self._cconn.send(None)
        break
      except Exception as e:
        tb = traceback.format_exc()
        self._cconn.send((e, tb))
        # raise e  # You can still rise this exception if you need to
        time.sleep(0.01)
        tries += 1

  @property
  def exception(self):
    if self._pconn.poll():
      self._exception = self._pconn.recv()
    return self._exception

def get_depth_image():
  """
  Function returning the most recent rgb, depth image
  """
  global depth_camera_connected, camera

  if not camera: return None, None

  if rgb_image is None or depth_image is None:
    rospy.logwarn("Warning: camera images are None")
  else:
    depth_camera_connected = True

  return rgb_image, depth_image

def rgb_callback(msg):
  """
  Save the most recent rgb image
  """
  global rgb_image
  cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
  rgb_image = np.array(cv_image, dtype=np.uint8)

def depth_callback(msg):
  """
  Save the most recent depth image
  """

  global depth_image
  cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
  depth_image = np.array(cv_image, dtype=np.float32)

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
    panda_z_height * -1 # simulation flips up/down
  ]

  # get panda estimated external wrenches
  if franka_instance is not None:
    local_wrench = franka_instance.get_end_effector_wrench_local()
    global_wrench = franka_instance.get_end_effector_wrench_global()

    if use_panda_ft_sensing:
      state.ftdata.force.x = local_wrench[0]
      state.ftdata.force.y = local_wrench[1]
      state.ftdata.force.z = local_wrench[2] * -1 # we want positive force to be upwards
      state.ftdata.torque.x = local_wrench[3]
      state.ftdata.torque.y = local_wrench[4]
      state.ftdata.torque.z = local_wrench[5]

  # optional: scale data
  global scale_gauge_data, scale_palm_data, scale_wrist_data
  state.sensor.gauge1 *= scale_gauge_data
  state.sensor.gauge2 *= scale_gauge_data
  state.sensor.gauge3 *= scale_gauge_data
  state.sensor.gauge4 *= scale_palm_data
  state.ftdata.force.z *= scale_wrist_data

  # if use a simulated ftsensor, override with simulated data
  if use_sim_ftsensor:
    state.ftdata.force.z = model.trainer.env.mj.sim_sensors.read_wrist_Z_sensor()

  # if we rezero ftsensor every time values repeat (ie connection lost)
  global last_ft_value
  if dynamic_recal_ftsensor:
    # rospy.loginfo(f"last value = {last_ft_value}, new value = {state.ftdata.force.z}")
    # if new value and old are exactly floating-point equal
    if state.ftdata.force.z == last_ft_value:
      model.trainer.env.mj.real_sensors.wrist_Z.offset = last_ft_value
      # if log_level > 1:
      #   rospy.loginfo("RECALIBRATED FTSENSOR TO ZERO")
    last_ft_value = state.ftdata.force.z

  # if we are trying to reject noise from the wrist sensor
  if reject_wrist_noise:
    rejection_threshold = 5
    new_reading = state.ftdata.force.z
    if last_ft_value is None:
      pass
    elif abs(state.ftdata.force.z - last_ft_value) > rejection_threshold:
      state.ftdata.force.z = last_ft_value
    last_ft_value = new_reading

  # assemble our state vector (order is vitally important! Check mjclass.cpp)
  sensor_vec = [
    state.sensor.gauge1, 
    state.sensor.gauge2, 
    state.sensor.gauge3, 
    state.sensor.gauge4,
    state.ftdata.force.z
  ]

  # TEMPORARY FIX - WILL NEED TO BE UPGRADED
  if use_devel:
    try:
      if model.trainer.env.params.XY_base_actions:
        state_vec.append(0)
        state_vec.append(0)
      if model.trainer.env.params.Z_base_rotation:
        state_vec.append(0)
    except Exception as e:
      pass

  # input the data
  model.trainer.env.mj.input_real_data(state_vec, sensor_vec)

  # prepare the image observation if the network is using them
  if model.trainer.env.params.use_rgb_in_observation:

    global image_observation
    image_observation = model.trainer.env._preprocess_real_image(rgb_image)
    # image_observation = image_observation.detach()

    # now revert the image to numpy unit8 from float [0, 1], since we want to visualise the transform
    rgb_np = (image_observation.clamp(-1.0, 1.0).cpu().float().numpy() + 1) / 2.0 * 255.0
    rgb_np = np.transpose(rgb_np, (0, 2, 1)) # rotate it so it comes out properly
    rgb_np = np.transpose(rgb_np, (2, 1, 0)).astype(np.uint8)

    # now create a ROS image message
    rgb_msg = bridge.cv2_to_imgmsg(rgb_np, encoding="rgb8")
    img_obs_pub.publish(rgb_msg)

  # publish the dqn network normalised input values
  global norm_state_pub, norm_sensor_pub
  norm_state = NormalisedState()
  norm_sensor = NormalisedSensor()
  local_wrench_msg = Wrench()
  global_wrench_msg = Wrench()

  # can visualise 'raw' data, 'SI' calibrated data, or 'normalised' network input data
  norm_state.gripper_x = model.trainer.env.mj.real_sensors.normalised.read_x_motor_position()
  norm_state.gripper_y = model.trainer.env.mj.real_sensors.normalised.read_y_motor_position()
  norm_state.gripper_z = model.trainer.env.mj.real_sensors.normalised.read_z_motor_position()
  norm_state.base_z = model.trainer.env.mj.real_sensors.normalised.read_z_base_position()

  norm_sensor.gauge1 = model.trainer.env.mj.real_sensors.normalised.read_finger1_gauge()
  norm_sensor.gauge2 = model.trainer.env.mj.real_sensors.normalised.read_finger2_gauge()
  norm_sensor.gauge3 = model.trainer.env.mj.real_sensors.normalised.read_finger3_gauge ()
  norm_sensor.palm = model.trainer.env.mj.real_sensors.normalised.read_palm_sensor()
  norm_sensor.wrist_z = model.trainer.env.mj.real_sensors.normalised.read_wrist_Z_sensor()

  SI_palm = model.trainer.env.mj.real_sensors.SI.read_palm_sensor()

  if franka_instance is not None:
    local_wrench_msg.force.x = local_wrench[0]
    local_wrench_msg.force.y = local_wrench[1]
    local_wrench_msg.force.z = local_wrench[2]
    local_wrench_msg.torque.x = local_wrench[3]
    local_wrench_msg.torque.y = local_wrench[4]
    local_wrench_msg.torque.z = local_wrench[5]

    global_wrench_msg.force.x = global_wrench[0]
    global_wrench_msg.force.y = global_wrench[1]
    global_wrench_msg.force.z = global_wrench[2]
    global_wrench_msg.torque.x = global_wrench[3]
    global_wrench_msg.torque.y = global_wrench[4]
    global_wrench_msg.torque.z = global_wrench[5]

    panda_local_wrench_pub.publish(local_wrench_msg)
    panda_global_wrench_pub.publish(global_wrench_msg)
  
  # if using a simulated ft sensor, replace with this data instead
  if use_sim_ftsensor:
    norm_sensor.wrist_z = model.trainer.env.mj.sim_sensors.read_wrist_Z_sensor()

  # if we are trying to prevent hitting the table
  if prevent_table_hit:
    global risk_table_hit
    if norm_sensor.wrist_z > 0.99 and norm_sensor.palm > 0.99:
      risk_table_hit = True
    else:
      risk_table_hit = False

  norm_state_pub.publish(norm_state)
  norm_sensor_pub.publish(norm_sensor)
  palm_SI_pub.publish(SI_palm)

  global grasp_frc_stable, stable_grasp_frc

  # determine if the grasp is stable
  gauge1 = model.trainer.env.mj.real_sensors.SI.read_finger1_gauge()
  gauge2 = model.trainer.env.mj.real_sensors.SI.read_finger2_gauge()
  gauge3 = model.trainer.env.mj.real_sensors.SI.read_finger3_gauge ()
  palm = model.trainer.env.mj.real_sensors.SI.read_palm_sensor()
  avg_gauge = (gauge1 + gauge2 + gauge3) / 3.0

  # check forces are within acceptable limits and the target height is reached
  if (avg_gauge > model.trainer.env.mj.set.stable_finger_force and
      palm > model.trainer.env.mj.set.stable_palm_force and
      avg_gauge < model.trainer.env.mj.set.stable_finger_force_lim and
      palm < model.trainer.env.mj.set.stable_palm_force_lim and
      panda_z_height > model.trainer.env.mj.set.gripper_target_height):
    force_str = f"Avg gauge = {avg_gauge:.1f} ({gauge1:.1f}, {gauge2:.1f}, {gauge3:.1f}), palm = {palm:.1f}"
    if log_level > 0 and stable_grasp_frc is None:
      rospy.loginfo(f"Stable grasp detected. {force_str}")
    stable_grasp_frc = [gauge1, gauge2, gauge3, palm]
    grasp_frc_stable = True

  # # latch the True message, reset to False in execute_grasping_callback
  # else:
  #   grasp_frc_stable = False

  global ready_for_new_action, action_received

  # once an action is completed, we need a new one
  if state.is_target_reached:

    global gripper_target_reached
    gripper_target_reached = True

def extra_gripper_data_callback(state):
  """
  Data callback if a second gripper is being used
  """

  global extra_gripper_palm_frc, extra_gripper_palm_raw

  raw_gauge = state.gauge4

  gauge_5N = 200000
  if raw_gauge > gauge_5N * 10 or raw_gauge < -gauge_5N * 10:
    return

  # gauge_cal_factor = -4.6199e-5 # december 9th calibration
  gauge_cal_factor = 2.787e-5 # feb 13th calibration (12V POWER SUPPLY)

  extra_gripper_palm_raw = raw_gauge * gauge_cal_factor 
  extra_gripper_palm_frc = extra_gripper_palm_raw - extra_gripper_z_offset

  extra_zfrc_pub.publish(extra_gripper_palm_frc)

def generate_action():
  """
  Get a new gripper state
  """

  global currently_testing
  global current_test_data

  obs = model.trainer.env.mj.get_real_observation()

  if debug_observation:
    rospy.loginfo("Debugging observation:")
    model.trainer.env.mj.debug_observation(obs)
    rospy.loginfo(f"SI values for sensors: {model.trainer.env.mj.get_simple_state_vector(model.trainer.env.mj.real_sensors.SI)}")

  if model.trainer.env.params.use_rgb_in_observation:
    torch_obs = model.trainer.env._make_img_obs(image_observation, obs)
  else:
    torch_obs = model.trainer.to_torch(obs)

  if print_network_eval_time:
    t1 = time.time()
  
  if currently_testing and current_test_data.data.heuristic:
    # if we are doing heuristic grasping
    action = model.trainer.env.get_heuristic_action()
  else:
    # use the state to predict the best action (test=True means pick best possible, decay_num has no effect)
    action = model.trainer.agent.select_action(torch_obs, decay_num=1, test=True)
    if len(action) == 1:
      action = (action.cpu()).item()
    else:
      action = (action.cpu()).numpy()

  if print_network_eval_time:
    t2 = time.time()
    rospy.loginfo(f"Time take for RL action selection {1e3 * (t2 - t1):.3f} milliseconds")

  if log_level >= 2: rospy.loginfo(f"Generated action, action code is: {action}")

  # # if we are preventing palm backwards actions
  # if prevent_back_palm and action == 5:
  #   if log_level > 0: rospy.loginfo(f"prevent_back_palm=TRUE, backwards palm prevented")
  #   action = 4

  # prevent x open
  if prevent_x_open and action == 1:
    if log_level > 0: rospy.loginfo(f"prevent_x_open=TRUE, setting Y close instead")
    action = 2

  # prevent the controller from hitting the table
  if prevent_table_hit and action == 6 and risk_table_hit:
    action = 7
    rospy.loginfo("Risk of table hit")

  # apply the action and get the new target state (vector)
  new_target_state = model.trainer.env._set_action(action)

  # if using a simulated ft sensor, resolve the action in simulation
  if use_sim_ftsensor or render_sim_view:
    if log_level > 0: rospy.loginfo("use_sim_ftsensor=TRUE, taking simulated action")
    model.trainer.env.mj.action_step()
    if render_sim_view: model.trainer.env.mj.render()

  # if at test time, save simple, unnormalised state data for this step
  if currently_testing:
    SI_state_vector = model.trainer.env.mj.get_simple_state_vector(model.trainer.env.mj.real_sensors.SI)
    current_test_data.add_step(obs, action, SI_vector=SI_state_vector)
    if quit_on_palm is not None:
      palm_force = model.trainer.env.mj.real_sensors.SI.read_palm_sensor()
      if palm_force > quit_on_palm:
        rospy.logwarn(f"PALM FORCE OF {palm_force:.1f} UNSAFE, CANCELLING GRASPING")
        cancel_grasping_callback()

  # determine if this action is for the gripper or panda
  if model.trainer.env.mj.last_action_gripper(): for_franka = False
  elif model.trainer.env.mj.last_action_panda(): for_franka = True
  else: raise RuntimeError("last action not on gripper or on panda")

  return new_target_state, for_franka

def move_panda_service(request):
  """
  Move the panda using a service
  """
  global franka_instance
  move_panda_z_abs(franka_instance, request.z_move_mm * 1e-3)

  return True

def move_panda_z_abs(franka, target_z):
  """
  Move the panda to a new z position with cartesian motion using Valerio's library,
  an instance of which is passed as 'franka'
  """
  
  global panda_in_motion
  while panda_in_motion:
    time.sleep(0.01)

  global panda_z_height

  # create identity matrix (must be floats!)
  T = np.array(
    [[1,0,0,0],
     [0,1,0,0],
     [0,0,1,0],
     [0,0,0,1]],
     dtype=np.float64
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
  global panda_z_move_duration

  panda_in_motion = True

  if use_panda_threads:
    process = Process(target=franka.move, args=("relative", T, panda_z_move_duration))
    process.start()
    process.join()
    process.close()
    if process.exception:
      rospy.logwarn(f"Panda movement exception, likely too high force on the table. Cancelling grasping. Error message: {process.exception}")
      cancel_grasping_callback()
  else:
    franka.move("relative", T, panda_z_move_duration)

  panda_in_motion = False

  # update with the new target position
  panda_z_height = target_z

  return

def run_extra_force_test(request=None):
  """
  Wrapper function to run a force test
  """

  global prevent_force_test
  if prevent_force_test:
    rospy.logwarn("run_extra_force_test stopped by user")
    prevent_force_test = False
    return 0.0

  axis = str(request.axis).lower()
  rospy.loginfo(f"run_extra_force_test() called: axis={axis}, object_num={request.obj_num}")

  if axis in ["x", "y"]:

    # move the panda to a hardcoded probe position
    if request.obj_num != 0:
      target_joints = hardcoded_object_force_measuring_ycb[str(request.obj_num)][axis.upper()]
      success = set_panda_joints(target_joints)
      if not success:
        # user or error has interrupted motion
        rospy.logwarn("run_extra_force_test() failed on panda motion, returning")
        connect_panda()
        return None

    # probe the force at this position
    max_force = test_force_with_extra_gripper()

    # reset the probe
    new_demand = GripperRequest()
    new_demand.message_type = "home"

    if log_level > 0: rospy.loginfo(f"Homing extra gripper following force test")
    extra_demand_pub.publish(new_demand)

  elif axis == "z":
    max_force = test_palm_force()

  else:
    max_force = 0.0
    rospy.logwarn(f"run_extra_force_test() warning: axis = {axis} not recognised")

  # report the result
  if max_force is None:
    rospy.logwarn("--- Force test FAILED, got None ---")
    return max_force
  
  rospy.logwarn(f"--- Force test axis: {axis} had force: {max_force:.1f}N ---")

  # save information to globals
  global trial_object_axis, axis_force_tol
  trial_object_axis = axis
  axis_force_tol = max_force

  return max_force

def test_force_with_extra_gripper(request=None):
  """
  Probe forwards with the palm of a 2nd gripper, in order to test the tolerated
  force. This assumes that the object and 2nd gripper are already in correct
  alignment before this function is called
  """

  if continue_demo:
    rospy.loginfo("No palm force test run, demo is active, returning")
    return 0.0

  global extra_gripper_measuring, XY_force_target, prevent_force_test

  rospy.loginfo("Running XY force test now")

  # # first move panda up to avoid table
  # time.sleep(0.1)
  # req = ResetPanda()
  # req.reset_height_mm = 80
  # reset_panda(req)

  if not extra_gripper_measuring: 
    rospy.logwarn("test_force_with_extra_gripper() called but extra_gripper_measuring=False")
    return 0.0

  start = time.time()
  max_time = 20
  palm_move = 5e-3
  palm_move_after_contact = 0
  max_palm_move = 150e-3
  max_palm_move_after_contact = 50e-3 # probe length
  force_for_contact = 0.5
  move_increment = 2e-3 # in metres

  force_target = XY_force_target
  start_force = extra_gripper_palm_frc
  max_tolerated_force = 0

  while time.time() - start < max_time and (not prevent_force_test):

    if True:

      # check if required force is reached
      new_force = extra_gripper_palm_frc
      tolerated_force = new_force - start_force
      if tolerated_force > max_tolerated_force: max_tolerated_force = tolerated_force
      if tolerated_force > force_target:
        rospy.logwarn(f"Palm test complete, tolerated {max_tolerated_force:.1f}N. Current force {new_force:.1f}, start force {start_force:.1f}, target force {force_target:.1f}")
        return max_tolerated_force
      
      # check if dangerous forces are recorded
      norm_gauge1 = model.trainer.env.mj.real_sensors.normalised.read_finger1_gauge()
      norm_gauge2 = model.trainer.env.mj.real_sensors.normalised.read_finger2_gauge()
      norm_gauge3 = model.trainer.env.mj.real_sensors.normalised.read_finger3_gauge ()
      norm_palm = model.trainer.env.mj.real_sensors.normalised.read_palm_sensor()
      norm_wrist_z = model.trainer.env.mj.real_sensors.normalised.read_wrist_Z_sensor()

      rospy.loginfo(f"Palm test in progress, just tolerated {tolerated_force:.1f}N, maximum tolerated {max_tolerated_force:.1f}. Current force {new_force:.1f}, start force {start_force:.1f}, target force {force_target:.1f}. (g1,g2,g3,p,w) = ({norm_gauge1:.2f}, {norm_gauge2:.2f}, {norm_gauge3:.2f}, {norm_palm:.2f}, {norm_wrist_z:.2f})")

      gauge_lim = 0.7 # sat yield factor is 1.5, hence yield is 1.0/1.5 = 0.67
      if norm_gauge1 > gauge_lim or norm_gauge2 > gauge_lim or norm_gauge3 > gauge_lim: # or norm_wrist_z > 0.3:
        rospy.logwarn(f"Palm test failed, safe sensors exceeded (g1,g2,g3,p,w) = ({norm_gauge1:.2f}, {norm_gauge2:.2f}, {norm_gauge3:.2f}, {norm_palm:.2f}, {norm_wrist_z:.2f}). Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force

      # if move_gripper is False:
      #   if log_level > 1: rospy.loginfo(f"Gripper action ignored as move_gripper=False")
      #   continue

      palm_move += move_increment

      if max_tolerated_force > force_for_contact:
        palm_move_after_contact += move_increment
      
      if palm_move > max_palm_move:
        rospy.logwarn(f"Palm test failed, maximum palm movement of {palm_move:.3f} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force
      
      if palm_move_after_contact > max_palm_move_after_contact:
        rospy.logwarn(f"Palm test failed, maximum palm movement after contact of {palm_move_after_contact:.3f} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force

      # send a demand to move the palm
      new_demand = GripperRequest()
      new_demand.x = 130e-3
      new_demand.y = 130e-3
      new_demand.z = palm_move
      new_demand.message_type = "command"

      if log_level > 0: rospy.loginfo(f"test_force_with_extra_gripper is send the palm to {palm_move * 1000:.1f} mm")
      extra_demand_pub.publish(new_demand)

      # sleep for it to be executed (hand tune this value)
      time.sleep(0.175)

  if prevent_force_test:
    rospy.logwarn("Force XY test stopped by user")
    prevent_force_test = False
  else:
    rospy.logwarn(f"Force XY test failed, timeout of {max_time} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")

  return max_tolerated_force

def test_palm_force():
  """
  Test grasp stability by applying extra palm force and seeing if the object remains in grasp
  """

  if continue_demo:
    rospy.loginfo("No palm force test run, demo is active, returning")
    return

  global gripper_target_reached, use_palm_force_test, palm_force_target, prevent_force_test

  rospy.loginfo("Running palm force test now")

  # first move panda up to avoid table
  time.sleep(0.1)
  req = ResetPanda()
  req.reset_height_mm = 80
  reset_panda(req)

  if not use_palm_force_test: 
    rospy.logwarn("test_palm_force() called but use_palm_force_test=False")
    return
  
  # get the current state position
  current_gripper_state = model.trainer.env.mj.get_state_metres(True) # realworld=True

  start = time.time()
  max_time = 30

  palm_move = 0e-3
  palm_move_after_contact = 0
  max_palm_move = 165e-3 - current_gripper_state[2]
  max_palm_move_after_contact = 50e-3
  force_for_contact = 0.5
  move_increment = 2e-3 # in metres

  force_target = palm_force_target
  start_force = model.trainer.env.mj.real_sensors.SI.read_palm_sensor()
  max_tolerated_force = 0

  while time.time() - start < max_time and not(prevent_force_test):

    if gripper_target_reached:

      # check if required force is reached
      new_force = model.trainer.env.mj.real_sensors.SI.read_palm_sensor()
      tolerated_force = new_force - start_force
      if tolerated_force > max_tolerated_force: max_tolerated_force = tolerated_force
      if tolerated_force > force_target:
        rospy.logwarn(f"Palm test complete, tolerated {max_tolerated_force:.1f}N. Current force {new_force:.1f}, start force {start_force:.1f}, target force {force_target:.1f}")
        return max_tolerated_force
      
      # check if dangerous forces are recorded
      norm_gauge1 = model.trainer.env.mj.real_sensors.normalised.read_finger1_gauge()
      norm_gauge2 = model.trainer.env.mj.real_sensors.normalised.read_finger2_gauge()
      norm_gauge3 = model.trainer.env.mj.real_sensors.normalised.read_finger3_gauge ()
      norm_palm = model.trainer.env.mj.real_sensors.normalised.read_palm_sensor()
      norm_wrist_z = model.trainer.env.mj.real_sensors.normalised.read_wrist_Z_sensor()

      rospy.loginfo(f"Palm test in progress, just tolerated {tolerated_force:.1f}N, maximum tolerated {max_tolerated_force:.1f}. Current force {new_force:.1f}, start force {start_force:.1f}, target force {force_target:.1f}. (g1,g2,g3,p,w) = ({norm_gauge1:.2f}, {norm_gauge2:.2f}, {norm_gauge3:.2f}, {norm_palm:.2f}, {norm_wrist_z:.2f})")

      gauge_lim = 0.7 # sat yield factor is 1.5, hence yield is 1.0/1.5 = 0.67
      if norm_gauge1 > gauge_lim or norm_gauge2 > gauge_lim or norm_gauge3 > gauge_lim or norm_wrist_z > 0.8:
        rospy.logwarn(f"Palm test failed, safe sensors exceeded (g1,g2,g3,p,w) = ({norm_gauge1:.2f}, {norm_gauge2:.2f}, {norm_gauge3:.2f}, {norm_palm:.2f}, {norm_wrist_z:.2f}). Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force

      if move_gripper is False:
        if log_level > 1: rospy.loginfo(f"Gripper action ignored as move_gripper=False")
        continue

      palm_move += move_increment

      if max_tolerated_force > force_for_contact:
        palm_move_after_contact += move_increment
      
      if palm_move > max_palm_move:
        rospy.logwarn(f"Palm test failed, maximum palm movement of {palm_move:.3f} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force
      
      if palm_move_after_contact > max_palm_move_after_contact:
        rospy.logwarn(f"Palm test failed, maximum palm movement after contact of {palm_move_after_contact:.3f} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")
        return max_tolerated_force

      # palm_move += 1e-3 # increment in metres
      
      # if palm_move > max_palm_move:
      #   rospy.logwarn(f"Palm test failed, maximum palm movement of {palm_move:.3f} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")
      #   return max_tolerated_force

      new_demand = GripperDemand()
      new_demand.state.pose.x = current_gripper_state[0]
      new_demand.state.pose.y = current_gripper_state[1]
      new_demand.state.pose.z = current_gripper_state[2] + palm_move

      if log_level > 0: rospy.loginfo("test_palm_force is publishing a new gripper demand")
      for i in range(number_gripper_demands):
        demand_pub.publish(new_demand)

      # data callback will let us know when the gripper demand is fulfilled
      gripper_target_reached = False

      # sleep for it to be executed (hand tune this value)
      time.sleep(0.175)

  if prevent_force_test:
    rospy.logwarn("Force test stopped by user")
    prevent_force_test = False
  else:
    rospy.logwarn(f"Palm test failed, timeout of {max_time} exceeded. Maximum tolerated force {max_tolerated_force:.1f}N")

  return max_tolerated_force

def execute_grasping_callback(request=None, reset=True):
  """
  Service callback to complete a dqn grasping task
  """

  if log_level > 0: rospy.loginfo("rl_grasping_node is now starting a grasping task")

  global ready_for_new_action
  global action_delay
  global continue_grasping
  global panda_z_height
  global step_num
  global panda_in_motion
  global gripper_target_reached
  global grasp_frc_stable
  global stable_grasp_frc
  global prevent_force_test

  if reset:
    reset_all(allow_panda_noise=True) # note this calls 'cancel_grasping_callback' if continue_grasping == True

  # set initial flags
  continue_grasping = True
  prevent_force_test = False
  grasp_frc_stable = False

  step_num = 0

  rate = rospy.Rate(50)

  while not rospy.is_shutdown() and continue_grasping:

    ready_for_new_action = (not panda_in_motion) * gripper_target_reached

    if ready_for_new_action and model_loaded:

      ready_for_new_action = False

      # are we ending grasps when forces fall in valid range
      # can visualise 'raw' data, 'SI' calibrated data, or 'normalised' network input data

      if use_forces_stable_grasp:

        if grasp_frc_stable:

          cancel_grasping_callback()

          # using a second gripper for XYZ force measurements
          if extra_gripper_measuring:

            # only automatically measure if doing a trial
            if trial_object_num is not None:

              global axis_force_tol
              rospy.loginfo(f"Trial in progress, extra_gripper_measuring=True. Trial number = {trial_object_num}, trial axis = {trial_object_axis}")
              test_req = ForceTest()
              test_req.axis = trial_object_axis
              test_req.obj_num = trial_object_num
              run_extra_force_test(test_req)

          # do a regular palm force test
          elif use_palm_force_test:
            test_palm_force()

          rospy.loginfo(f"Stable grasp forces were: {stable_grasp_frc}")

        # grasp forces are not yet stable, continue
        else:
          # true message latches whilst we execute the action, now reset
          grasp_frc_stable = False
          stable_grasp_frc = None

      # not using forces for stable grasp, check only height
      else:

        # if we have reached our target position, terminate grasping
        if panda_z_height > panda_target_height_mm * 1e-3 - 1e-6:
          if log_level > 0: rospy.loginfo("Panda height has reached 30mm, stopping grasping")
          cancel_grasping_callback()

      # if we have reached our step limit, terminate grasping
      if step_num > model.trainer.env.params.max_episode_steps:
        if log_level > 0: rospy.loginfo(f"Max episode steps of {model.trainer.env.params.max_episode_steps} exceeded, stopping grasping")
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
      if model.trainer.env.using_continous_actions() or not for_franka:

        if move_gripper is False:
          if log_level > 1: rospy.loginfo(f"Gripper action ignored as move_gripper=False")
          continue

        new_demand = GripperDemand()
        new_demand.state.pose.x = new_target_state[0]
        new_demand.state.pose.y = new_target_state[1]
        new_demand.state.pose.z = new_target_state[2]

        if log_level > 0: rospy.loginfo("rl_grasping_node is publishing a new gripper demand")
        for i in range(number_gripper_demands):
          demand_pub.publish(new_demand)

        # data callback will let us know when the gripper demand is fulfilled
        gripper_target_reached = False

      # if the action is for the panda
      if model.trainer.env.using_continous_actions() or for_franka:

        if move_panda is False:
          if log_level > 1: rospy.loginfo(f"Panda action ignored as move_panda=False")
          continue

        if model.trainer.env.params.XY_base_actions:
          panda_x = new_target_state[3]
          panda_y = new_target_state[4]
          panda_z = -new_target_state[5]
          panda_target = panda_z
          panda_target = -new_target_state[5] # negative/positive flipped
        else:
          panda_target = -new_target_state[3] # negative/positive flipped
        if log_level > 1: rospy.loginfo(f"dqn is sending a panda control signal to z = {panda_target:.6f}")
        move_panda_z_abs(franka_instance, panda_target)

        # move panda is blocking, so we know we can now have a new action
        # ready_for_new_action = True

    rate.sleep()

  rospy.loginfo("Leaving execute_grasping_callback()")

  return []

def cancel_grasping_callback(request=None):
  """
  Callback to end a grasping task, does not stop immediately
  """

  rospy.loginfo("Cancelling grasping now")

  global continue_grasping, grasp_frc_stable
  continue_grasping = False
  grasp_frc_stable = False 

  return []

def stop_force_test(request=None):
  """
  Prevent a force test from completing
  """

  global prevent_force_test
  prevent_force_test = True

  return []

def reset_all(request=None, skip_panda=False, allow_panda_noise=False):
  """
  Reset the gripper and panda
  """

  # ensure grasping is cancelled
  if continue_grasping:
    cancel_grasping_callback()

  # reset the gripper position (not blocking, publishes request)
  if move_gripper:
    if log_level > 1: rospy.loginfo("rl_grasping_node is about to try and reset gripper position")
    reset_gripper()
  elif log_level > 1: rospy.loginfo("rl_grasping_node not reseting gripper position as move_gripper is False")

  # reset the panda position (blocking)
  if move_panda and not skip_panda:
    if log_level > 1: rospy.loginfo(f"rl_grasping_node is about to try and reset panda position to {panda_reset_height_mm}")
    panda_reset_req = ResetPanda()
    panda_reset_req.reset_height_mm = panda_reset_height_mm # dqn episode begins at 10mm height
    time.sleep(0.2) # give time for palm to move in case we will lower the panda into the object
    reset_panda(panda_reset_req, allow_noise=allow_panda_noise)
  elif log_level > 1: rospy.loginfo("rl_grasping_node not reseting panda position as move_panda is False")

  # reset and prepare environment
  rospy.sleep(2.0)  # sleep to allow sensors to settle before recalibration

  # reset the environment for real world testing
  model.trainer.env.reset(realworld=True) # recalibrates sensors

  if extra_gripper_measuring:
    # manually reset offset to rezero the sensor
    global extra_gripper_z_offset
    extra_gripper_z_offset = extra_gripper_palm_raw

  if currently_testing and current_test_data.data.heuristic:
    model.trainer.env.start_heuristic_grasping(realworld=True)

  # check for settings clashes
  if use_sim_ftsensor and dynamic_recal_ftsensor:
      if log_level > 0: 
        rospy.logwarn("use_sim_ftsensor and dynamic_recal_ft_sensor both True at the same time")

  if log_level > 1: rospy.loginfo("rl_grasping_node reset_all() is finished, sensors recalibrated")

  return []

def reset_panda(request=None, allow_noise=False):
  """
  Reset the panda position to a hardcoded joint state.
  """

  def height_to_cal(height):
    """
    Converts an integer height to the nearest calibration 2mm calibration height
    """
    multiple_of_2 = 2 * round(height/2)
    if multiple_of_2 < 0: raise RuntimeError(f"height_to_call got multiple_of_2 below zero, input height = {height}")
    return f"cal_{multiple_of_2:.0f}_mm"

  global franka_instance
  global move_panda
  global log_level
  global panda_z_height
  global photoshoot_calibration

  if not move_panda:
    if log_level > 0: rospy.logwarn("asked to reset_panda() but move_panda is false")
    return False
  
  global panda_in_motion
  while panda_in_motion:
    time.sleep(0.01)

  if photoshoot_calibration:

    # calibrations 31/5/23 for side on videos (tie palm cable back), 0=v. firm neoprene press
    cal_0_mm = [-0.10402882, 0.50120518, 0.10778944, -1.01105512, -0.04686748, 1.50164708, -1.26073515]
    cal_2_mm = [-0.10387006, 0.50527945, 0.10769805, -1.00157544, -0.04692841, 1.49511798, -1.26073252]
    cal_4_mm = [-0.10330050, 0.50944433, 0.10765647, -0.99195720, -0.04720724, 1.48748527, -1.26073653]
    cal_6_mm = [-0.10260091, 0.51340940, 0.10745183, -0.98116904, -0.04766048, 1.47974774, -1.26075086]
    cal_10_mm = [-0.10159385, 0.52105691, 0.10745041, -0.95851852, -0.04829950, 1.46465763, -1.26075028]
    cal_12_mm = [-0.10102108, 0.52506261, 0.10745403, -0.94637724, -0.04867820, 1.45684754, -1.26076261]
    cal_14_mm = [-0.10039027, 0.52930519, 0.10745403, -0.93433454, -0.04912472, 1.44883549, -1.26076231]
    cal_16_mm = [-0.09971443, 0.53378781, 0.10745403, -0.92148019, -0.04960671, 1.44069648, -1.26076865]
    cal_20_mm = [-0.09815959, 0.54329784, 0.10745573, -0.89504912, -0.05062444, 1.42390813, -1.26081059]
    cal_30_mm = [-0.09425711, 0.57439487, 0.10747072, -0.82226615, -0.05352376, 1.37814662, -1.26075268]
    cal_40_mm = [-0.08839450, 0.61583667, 0.10746761, -0.72435514, -0.05777517, 1.32181706, -1.26131619]
    cal_50_mm = [-0.07592334, 0.70676842, 0.10747491, -0.53616111, -0.06762633, 1.22484667, -1.26360272]

  else:

    # new calibrations 75DEGREE FINGERS BARE TABLE (NO WRIST) 4mm=light touch, 2mm=press
    cal_0_mm = [-0.27387776, -0.02585006, 0.30729622, -1.89643083, 0.00676038, 1.87621824, 0.84037356]
    cal_2_mm = [-0.27392039, -0.02572424, 0.30729622, -1.89337982, 0.00672408, 1.87142800, 0.84036733]
    cal_4_mm = [-0.27400394, -0.02574353, 0.30729093, -1.88897988, 0.00673235, 1.86650059, 0.84031136]
    cal_6_mm = [-0.27421166, -0.02623554, 0.30729463, -1.88446972, 0.00674949, 1.86159163, 0.84021751]
    cal_8_mm = [-0.27424804, -0.02660902, 0.30729630, -1.88004050, 0.00677693, 1.85652403, 0.84012095]
    cal_10_mm = [-0.27428154, -0.02698982, 0.30729559, -1.87545015, 0.00681192, 1.85143192, 0.84002916]
    cal_12_mm = [-0.27454526, -0.02738393, 0.30730011, -1.87072350, 0.00689858, 1.84639242, 0.83994788]
    cal_14_mm = [-0.27458083, -0.02776590, 0.30729819, -1.86601039, 0.00700295, 1.84137630, 0.83987414]
    cal_16_mm = [-0.27461111, -0.02810298, 0.30729511, -1.86128947, 0.00710709, 1.83632393, 0.83981037]
    cal_18_mm = [-0.27463627, -0.02841951, 0.30729817, -1.85654005, 0.00718308, 1.83133528, 0.83975061]
    cal_20_mm = [-0.27467913, -0.02870156, 0.30729826, -1.85180733, 0.00724723, 1.82633607, 0.83968904]
    cal_22_mm = [-0.27492752, -0.02881564, 0.30730066, -1.84703765, 0.00730904, 1.82137303, 0.83964258]
    cal_24_mm = [-0.27495433, -0.02910731, 0.30729987, -1.84214418, 0.00736595, 1.81627148, 0.83959877]
    cal_26_mm = [-0.27498185, -0.02917162, 0.30729728, -1.83711943, 0.00740111, 1.81114207, 0.83956440]
    cal_28_mm = [-0.27500193, -0.02946890, 0.30729693, -1.83219466, 0.00744732, 1.80599850, 0.83953597]
    cal_30_mm = [-0.27501834, -0.02951123, 0.30730128, -1.82733378, 0.00747141, 1.80099033, 0.83951795]
    cal_40_mm = [-0.27535423, -0.02973620, 0.30729625, -1.80185362, 0.00748783, 1.77566828, 0.83950376]
    cal_50_mm = [-0.27537779, -0.02964236, 0.30729646, -1.77550289, 0.00747043, 1.75009969, 0.83951632]
    cal_60_mm = [-0.27538812, -0.02767563, 0.30729625, -1.74815756, 0.00737045, 1.72428787, 0.83968787]
    cal_70_mm = [-0.27534187, -0.02516956, 0.30729125, -1.71972291, 0.00653091, 1.69838691, 0.84020591]
    cal_80_mm = [-0.27489364, -0.02177300, 0.30729093, -1.69023271, 0.00559191, 1.67217843, 0.84087814]
    cal_90_mm = [-0.27442663, -0.01700015, 0.30725260, -1.66094411, 0.00428391, 1.64610852, 0.84159472]
    cal_100_mm = [-0.27342677, -0.01169027, 0.30724883, -1.62902436, 0.00271029, 1.61938627, 0.84278577]
    cal_110_mm = [-0.27221279, -0.00529769, 0.30724584, -1.59589389, 0.00080301, 1.59226100, 0.84416299]
    cal_120_mm = [-0.27068688, 0.00206569, 0.30723810, -1.56146977, -0.00141168, 1.56471790, 0.84578688]
    cal_130_mm = [-0.26867866, 0.01051012, 0.30723448, -1.52532189, -0.00398714, 1.53676718, 0.84757781]
    cal_140_mm = [-0.26635378, 0.02023801, 0.30722737, -1.48763902, -0.00694801, 1.50828884, 0.84963912]

    # # calibration 15/02/24 60DEGREE FINGERS EI1 0mm=press, 2mm=light touch
    # cal_0_mm = [-0.26363617, -0.05473505, 0.30020690, -1.90742241, 0.00734889, 1.87581724, 0.82313380]
    # cal_2_mm = [-0.26366258, -0.05472722, 0.30021328, -1.90369823, 0.00735250, 1.87093787, 0.82305917]
    # cal_4_mm = [-0.26371654, -0.05471687, 0.30021328, -1.89936716, 0.00741290, 1.86591738, 0.82294415]
    # cal_6_mm = [-0.26391464, -0.05504486, 0.30021328, -1.89472156, 0.00748206, 1.86090533, 0.82281917]
    # cal_8_mm = [-0.26396259, -0.05532080, 0.30021161, -1.88997358, 0.00757329, 1.85586071, 0.82272108]
    # cal_10_mm = [-0.26403608, -0.05563658, 0.30021328, -1.88528073, 0.00770713, 1.85078742, 0.82262177]
    # cal_12_mm = [-0.26422203, -0.05593439, 0.30021523, -1.88056663, 0.00779554, 1.84573231, 0.82252450]
    # cal_14_mm = [-0.26427300, -0.05619965, 0.30021331, -1.87574604, 0.00784266, 1.84068288, 0.82243819]
    # cal_16_mm = [-0.26438289, -0.05632581, 0.30021690, -1.87089863, 0.00788102, 1.83561542, 0.82236489]
    # cal_18_mm = [-0.26453586, -0.05657995, 0.30021523, -1.86600990, 0.00791936, 1.83057723, 0.82228960]
    # cal_20_mm = [-0.26458381, -0.05668035, 0.30021328, -1.86109418, 0.00795603, 1.82553612, 0.82223020]
    # cal_22_mm = [-0.26463690, -0.05691966, 0.30021328, -1.85616508, 0.00798505, 1.82048298, 0.82216497]
    # cal_24_mm = [-0.26482269, -0.05695218, 0.30022009, -1.85119037, 0.00800458, 1.81538102, 0.82211627]
    # cal_26_mm = [-0.26486016, -0.05698160, 0.30021650, -1.84621390, 0.00801417, 1.81027087, 0.82207689]
    # cal_28_mm = [-0.26489700, -0.05699540, 0.30021845, -1.84108635, 0.00801764, 1.80515880, 0.82204296]
    # cal_30_mm = [-0.26493536, -0.05700533, 0.30021845, -1.83597784, 0.00801835, 1.80008457, 0.82202409]
    # cal_40_mm = [-0.26529586, -0.05696333, 0.30022013, -1.80990437, 0.00799617, 1.77460503, 0.82199937]
    # cal_50_mm = [-0.26544880, -0.05556902, 0.30021846, -1.78293351, 0.00772567, 1.74892897, 0.82204890]
    # cal_60_mm = [-0.26546751, -0.05339131, 0.30021507, -1.75493735, 0.00707560, 1.72307920, 0.82238232]
    # cal_70_mm = [-0.26539159, -0.05028930, 0.30021145, -1.72586789, 0.00612781, 1.69704613, 0.82291076]
    # cal_80_mm = [-0.26503468, -0.04624956, 0.30021437, -1.69573253, 0.00491428, 1.67075798, 0.82362430]
    # cal_90_mm = [-0.26424438, -0.04132709, 0.30020633, -1.66449699, 0.00342211, 1.64419101, 0.82454922]
    # cal_100_mm = [-0.26334276, -0.03539899, 0.30020896, -1.63179003, 0.00161609, 1.61728759, 0.82568431]
  
  # umh = 4 # uncertainty matrix height
  calibrated_0mm = 2# + umh # height in mm where table touch occurs with above calibration

  # beyond this we do not have additional 2mm increments
  if request.reset_height_mm >= 28: calibrated_0mm = 0

  # calculate the desired reset height
  if request is None:
    reset_to = 10 # default, panda is limited to +-30mm in move_panda_z_abs
  else:
    reset_to = request.reset_height_mm
  reset_to += calibrated_0mm
  if panda_reset_noise_mm is not None and allow_noise:
    reset_to += (random.random() * 2 * panda_reset_noise_mm) - panda_reset_noise_mm

  target_state_name = height_to_cal(reset_to)
  rospy.loginfo(f"reset_panda() will reset to a height given by '{target_state_name}'")

  # assign this reset height to a state vector from above (eg 'cal_10_mm')
  try:
    target_state = eval(f"{target_state_name}")
  except NameError as e:
    rospy.logerr(f"reset_panda() error: desired target height has no calibrated value")
    raise RuntimeError(f"reset_panda() error: {e}")

  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.1 # 0.1 is slow movements

  panda_in_motion = True

  if use_panda_threads:
    process = Process(target=franka_instance.move_joints, args=(target_state, speed_factor))
    process.start()
    process.join()
    process.close()
    if process.exception:
      rospy.logwarn(f"Panda movement exception, likely too high force on the table. Cancelling grasping. Error message: {process.exception}")
      cancel_grasping_callback()
  else:
    franka_instance.move_joints(target_state, speed_factor) # now approach reset height

  panda_in_motion = False

  panda_z_height = 0

  if log_level > 0: rospy.loginfo(f"panda reset to a calibrated height of {2 * round(reset_to / 2) - calibrated_0mm} (rounded from {reset_to:.1f} and calibrated zero of {calibrated_0mm})")

  return True

def set_panda_joints(target_state):
  """
  Set the panda joints to a particular state
  """

  rospy.loginfo(f"About to set panda to target joints: {target_state}")

  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.1 # 0.1 is slow movements, 0.05 very slow

  # to prevent 'attempted to start multiple motions'
  time.sleep(0.1)

  global panda_in_motion
  panda_in_motion = True

  if use_panda_threads:
    process = Process(target=franka_instance.move_joints, args=(target_state, speed_factor))
    process.start()
    process.join()
    process.close()
    if process.exception:
      rospy.logwarn(f"Panda movement exception. Cancelling grasping. Error message: {process.exception}")
      cancel_grasping_callback()
      return False
  else:
    franka_instance.move_joints(target_state, speed_factor) # now approach reset height

  panda_in_motion = False

  rospy.loginfo("Panda set to the target joint state")

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

  if log_level > 0: rospy.loginfo("rl_grasping_node requests homing gripper position")
  demand_pub.publish(homing_demand)

  if extra_gripper_measuring:
    home_request = GripperRequest()
    home_request.message_type = "home"
    extra_demand_pub.publish(home_request)

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

  if log_level > 1: rospy.loginfo("rl_grasping_node requesting gripper debug information")

  req = GripperInput()
  req.print_debug = True
  debug_pub.publish(req)

  return []

def load_model(request):
  """
  Load a model. Note if 'best_id' is True then the TrainingMananger defaults will
  select the stage. In most cases stage=max, hence the best training will be selected
  from the last curriculum stage
  """
  
  global model_loaded
  global model, dqn_log_level

  model = TrainingManager(log_level=dqn_log_level, device=device)

  if request.timestamp != "" and request.job_number != 0:
    use_timestamp_job_number = True
  else:
    use_timestamp_job_number = False

  # apply defaults
  if request.folderpath == "":
    request.folderpath = mymujoco_rl_path + "/models/" #"/home/luke/mymujoco/rl/models/"
  if request.run_id == 0:
    request.run_id = None
    best_id = False
  elif request.run_id == "best":
    request.run_id = None
    best_id = True
  else:
    best_id = False

  try:

    if use_timestamp_job_number:
      model.set_group_run_name(job_num=request.job_number, timestamp=request.timestamp)
      run_name = model.run_name
      pathtofolder = request.folderpath + "/" + model.group_name + "/"
    else:
      run_name = request.run_name
      pathtofolder = request.folderpath + "/" + request.group_name + "/"
      
    if log_level > 0: rospy.loginfo(f"Preparing to load model now in rl_grasping_node: {pathtofolder}/{run_name}")
    model.load(run_name=run_name, path_to_run_folder=pathtofolder, id=request.run_id, use_abs_path=True, best_id=best_id)
    model_loaded = True
    if log_level > 0: rospy.loginfo("Model loaded successfully")

    model.trainer.env.mj.set.set_use_noise(False) # disable all noise
    model.trainer.env.reset()

    # IMPORTANT: have to run calibration function to setup real sensors
    model.trainer.env.mj.calibrate_real_sensors()

    if abs(scale_actions_on_load - 1.0) > 1e-5:
      rospy.logwarn(f"Model actions are being scaled by {scale_actions_on_load}")
      model.trainer.env.mj.set.gripper_prismatic_X.value *= scale_actions_on_load
      model.trainer.env.mj.set.gripper_revolute_Y.value *= scale_actions_on_load
      model.trainer.env.mj.set.gripper_Z.value *= scale_actions_on_load
      model.trainer.env.mj.set.base_X.value *= scale_actions_on_load
      model.trainer.env.mj.set.base_Y.value *= scale_actions_on_load
      model.trainer.env.mj.set.base_Z.value *= scale_actions_on_load
      to_print = "New action values are:\n"
      to_print += f"gripper_prismatic_X = {model.trainer.env.mj.set.gripper_prismatic_X.value}\n"
      to_print += f"gripper_revolute_Y = {model.trainer.env.mj.set.gripper_revolute_Y.value}\n"
      to_print += f"gripper_Z = {model.trainer.env.mj.set.gripper_Z.value}\n"
      to_print += f"base_X = {model.trainer.env.mj.set.base_X.value}\n"
      to_print += f"base_Y = {model.trainer.env.mj.set.base_Y.value}\n"
      to_print += f"base_Z = {model.trainer.env.mj.set.base_Z.value}\n"
      rospy.logwarn(to_print)

    if abs(scale_gauge_data - 1.0) > 1e-5:
      rospy.logwarn(f"Gauge data is being scaled by: {scale_gauge_data}")
    if abs(scale_wrist_data - 1.0) > 1e-5:
      rospy.logwarn(f"Wrist data is being scaled by: {scale_wrist_data}")
    if abs(scale_palm_data - 1.0) > 1e-5:
      rospy.logwarn(f"Palm data is being scaled by: {scale_palm_data}")

    return True

  except Exception as e:

    rospy.logerr(e)
    rospy.logerr("Failed to load model in rl_grasping_node")

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

  global testsaver, current_test_data, model, currently_testing, image_rate

  if log_level > 0: rospy.loginfo(f"Starting a new test with name: {request.name}")

  if current_test_data is not None:
    if request.name == current_test_data.data.test_name:
      rospy.logwarn("Test already running with this name, aborted. Run end_test() first")
      return []

  # see if the camera is working (this function records if images are succesfully received)
  get_depth_image()

  if not depth_camera_connected and camera:
    rospy.logwarn("Depth camera is not connected, test aborted")
    return False
  elif not depth_camera_connected and not camera:
    rospy.logwarn("Depth camera set to FALSE, test proceeds")

  testsaver.enter_folder(request.name, forcecreate=True)
  current_test_data = GraspTestData() # wipe data clean
  current_test_data.start_test(request.name, model, get_depth_image, image_rate=image_rate)
  currently_testing = True

  # save the model in use if this file does not exist already
  savepath = testsaver.get_current_path()
  if not os.path.exists(savepath + "Tracking_info" + testsaver.file_ext()):
    model.trainer.modelsaver = testsaver
    model.trainer.save()
  else:
    if log_level > 1:
      rospy.loginfo("start_test() is not saving model as it already exists")

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

def save_test(request=None):
  """
  Save test data from a test
  """

  global current_test_data

  if log_level > 0: 
    rospy.loginfo(f"Saving test with name: {current_test_data.data.test_name}")
  if log_level > 1: rospy.loginfo(f"Saving test data now...")
  
  testsaver.save("test_data", pyobj=current_test_data.data)

  if log_level > 1: rospy.loginfo("...finished saving test data")

  return []

def end_test(request):
  """
  Finish a grasping test
  """

  global current_test_data, currently_testing

  if log_level > 0: 
    rospy.loginfo(f"Ending test with name: {current_test_data.data.test_name}")

  save_test()

  currently_testing = False
  current_test_data = None

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

  global trial_object_num, trial_object_axis
  trial_object_num = request.object_number
  trial_object_axis = str(request.axis)

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

  # special behaviour for automatically filling in force tolerated values, wipe these variables below!
  global trial_object_axis, trial_object_num, axis_force_tol
  if extra_gripper_measuring:
    if request.s_h:
      if trial_object_axis is not None and axis_force_tol is not None:
        if trial_object_axis.lower() == "x" and request.XFTol < 1e-4:
          rospy.logwarn(f"Auto recording X AXIS force as {axis_force_tol:.1f}N")
          request.XFTol = axis_force_tol
        elif trial_object_axis.lower() == "y" and request.YFTol < 1e-4:
          rospy.logwarn(f"Auto recording Y AXIS force as {axis_force_tol:.1f}N")
          request.YFTol = axis_force_tol
        elif trial_object_axis.lower() == "z" and request.pFTol < 1e-4:
          rospy.logwarn(f"Auto recording Z AXIS force as {axis_force_tol:.1f}N")
          request.pFTol = axis_force_tol
      rospy.loginfo(f"Tolerated forces: X={request.XFTol:.1f} Y={request.YFTol:.1f} Z={request.pFTol:.1f}")
    else:
      rospy.loginfo("Tolerated forces: NONE as stable height = False")

  global current_test_data
  current_test_data.finish_trial(request, stable_forces=stable_grasp_frc)

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

  # wipe local data, added for XYZ force measurement (marked as global above!)
  trial_object_num = None
  trial_object_axis = None
  axis_force_tol = None

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

  load = LoadModel()
  load.folderpath = "/home/luke/mujoco-devel/rl/models"
  load.run_id = "best"

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
      load.timestamp = "13-12-23_17-23"
      load.job_number = 2

    elif request.sensors == 1:
      load.timestamp = "13-12-23_17-23"
      load.job_number = 17

    elif request.sensors == 2:
      load.timestamp = "13-12-23_17-27"
      load.job_number = 42

    elif request.sensors == 3:
      load.timestamp =  "08-12-23_19-19"
      load.job_number = 53

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 1:

    rospy.loginfo("Loading model for finger 1.0mm thick and 24mm width")

    if request.sensors == 0:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 1:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 2:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 3:
      load.timestamp = "11-12-23_19-57"
      load.job_number = 120

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_new_model()")

  elif finger == 2:

    rospy.loginfo("Loading model for finger 1.0mm thick and 28mm width")

    if request.sensors == 0:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 1:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 2:
      load.timestamp = None
      load.job_number = None

    elif request.sensors == 3:
      load.timestamp = "11-12-23_19-57"
      load.job_number = 177

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_new_model()")

  rospy.loginfo(f"Number of sensors is {request.sensors}")
  rospy.loginfo(f"Timestamp: {load.timestamp}, job number: {load.job_number}")

  if load.timestamp is None or load.job_number is None:
    rospy.logerr("Requested baseline model not yet implemented")
    return False

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

def set_reject_wrist_noise_callback(request):
  """
  Set the bool value of whether we dynamically reset the ftsensor to zero
  every time we get two exactly repeated values (indicating a connection drop,
  which implies the sensor is in steady state and under no force)
  """

  global reject_wrist_noise
  if log_level > 1: rospy.loginfo(f"reject_wrist_noise originally set to {reject_wrist_noise}")
  reject_wrist_noise = request.data
  if log_level > 0: rospy.loginfo(f"reject_wrist_noise is now set to {reject_wrist_noise}")

  return []

def make_test_heurisitc(request=None):
  """
  Set the given test to be heuristic grasping
  """
  global current_test_data
  if log_level > 1: rospy.loginfo(f"test heuristic was set to {current_test_data.data.heuristic}")
  current_test_data.data.heuristic = True
  global model
  model.trainer.env.mj.set.continous_actions = False
  if log_level > 0: rospy.loginfo(f"test heurisitc now set to TRUE")

  return []

def gentle_place(request=None):
  """
  Place an object gently below the gripper
  """

  # ensure there is no graspng going on
  cancel_grasping_callback()

  # what is the current gripper position
  [gx, gy, gz, h] = model.trainer.env.mj.get_state_metres(True) # realworld=True for real data

  # # move the panda to a lower height
  # panda_reset_req = ResetPanda()
  # panda_reset_req.reset_height_mm = 20 # 10mm
  # reset_panda(panda_reset_req)

  # send a command to the gripper to open the fingers
  gy = gx + 5e-3
  if gy > 133.9e-3: 
    gx = 128.0e-3
    gy = 133.0e-3
  new_demand = GripperDemand()
  new_demand.state.pose.x = gx
  new_demand.state.pose.y = gy
  new_demand.state.pose.z = 5e-3

  if log_level > 0: rospy.loginfo("rl_grasping_node is publishing gripper demand for gentle place")
  demand_pub.publish(new_demand)

  time.sleep(1)

  while not rospy.is_shutdown():

    if not gripper_target_reached:
      time.sleep(0.05)
    else: break

  time.sleep(1)

  reset_gripper()

def move_panda_to(request=None):
  """
  Move the panda to a known joint position. These joint positions should
  be calibrated and tested by hand, do NOT run this function with untested
  joint positions
  """

  def height_to_cal(height):
    """
    Converts an integer height to the nearest calibration 2mm calibration height
    """
    multiple_of_2 = 2 * round(height/2)
    if multiple_of_2 < 0: raise RuntimeError(f"height_to_call got multiple_of_2 below zero, input height = {height}")
    return f"cal_{multiple_of_2:.0f}_mm"

  global move_panda, panda_z_height

  if not move_panda:
    rospy.logwarn("move_panda_to() aborted as move_panda = False")
    return False

  if request.move_to_int == 0:
    # regular calibrated spot (right side of diamond)
    cal_0_mm = [-0.27387776, -0.02585006, 0.30729622, -1.89643083, 0.00676038, 1.87621824, 0.84037356]
    cal_2_mm = [-0.27392039, -0.02572424, 0.30729622, -1.89337982, 0.00672408, 1.87142800, 0.84036733]
    cal_4_mm = [-0.27400394, -0.02574353, 0.30729093, -1.88897988, 0.00673235, 1.86650059, 0.84031136]
    cal_6_mm = [-0.27421166, -0.02623554, 0.30729463, -1.88446972, 0.00674949, 1.86159163, 0.84021751]
    cal_8_mm = [-0.27424804, -0.02660902, 0.30729630, -1.88004050, 0.00677693, 1.85652403, 0.84012095]
    cal_10_mm = [-0.27428154, -0.02698982, 0.30729559, -1.87545015, 0.00681192, 1.85143192, 0.84002916]
    cal_12_mm = [-0.27454526, -0.02738393, 0.30730011, -1.87072350, 0.00689858, 1.84639242, 0.83994788]
    cal_14_mm = [-0.27458083, -0.02776590, 0.30729819, -1.86601039, 0.00700295, 1.84137630, 0.83987414]
    cal_16_mm = [-0.27461111, -0.02810298, 0.30729511, -1.86128947, 0.00710709, 1.83632393, 0.83981037]
    cal_18_mm = [-0.27463627, -0.02841951, 0.30729817, -1.85654005, 0.00718308, 1.83133528, 0.83975061]
    cal_20_mm = [-0.27467913, -0.02870156, 0.30729826, -1.85180733, 0.00724723, 1.82633607, 0.83968904]
    cal_22_mm = [-0.27492752, -0.02881564, 0.30730066, -1.84703765, 0.00730904, 1.82137303, 0.83964258]
    cal_24_mm = [-0.27495433, -0.02910731, 0.30729987, -1.84214418, 0.00736595, 1.81627148, 0.83959877]
    cal_26_mm = [-0.27498185, -0.02917162, 0.30729728, -1.83711943, 0.00740111, 1.81114207, 0.83956440]
    cal_28_mm = [-0.27500193, -0.02946890, 0.30729693, -1.83219466, 0.00744732, 1.80599850, 0.83953597]
    cal_30_mm = [-0.27501834, -0.02951123, 0.30730128, -1.82733378, 0.00747141, 1.80099033, 0.83951795]
    cal_40_mm = [-0.27535423, -0.02973620, 0.30729625, -1.80185362, 0.00748783, 1.77566828, 0.83950376]
    cal_50_mm = [-0.27537779, -0.02964236, 0.30729646, -1.77550289, 0.00747043, 1.75009969, 0.83951632]
    cal_60_mm = [-0.27538812, -0.02767563, 0.30729625, -1.74815756, 0.00737045, 1.72428787, 0.83968787]
    cal_70_mm = [-0.27534187, -0.02516956, 0.30729125, -1.71972291, 0.00653091, 1.69838691, 0.84020591]
    cal_80_mm = [-0.27489364, -0.02177300, 0.30729093, -1.69023271, 0.00559191, 1.67217843, 0.84087814]

  elif request.move_to_int == 1:
    # back most point (top of diamond)
    cal_0_mm = [-0.28065882, -0.69304390, -0.03156822, -2.51254180, -0.02046181, 1.81425078, 0.34286978]
    cal_2_mm = [-0.28065837, -0.69289453, -0.03154801, -2.50850871, -0.02045031, 1.80923233, 0.34287310]
    cal_4_mm = [-0.28061365, -0.69290636, -0.03154535, -2.50364990, -0.02044683, 1.80406595, 0.34289447]
    cal_6_mm = [-0.28037703, -0.69293501, -0.03154002, -2.49871469, -0.02043928, 1.79894195, 0.34292478]
    cal_8_mm = [-0.28033246, -0.69316666, -0.03154002, -2.49369913, -0.02042766, 1.79382667, 0.34295695]
    cal_10_mm = [-0.28027092, -0.69321108, -0.03153835, -2.48873954, -0.02041460, 1.78875798, 0.34300284]
    cal_12_mm = [-0.28004255, -0.69324148, -0.03153683, -2.48366155, -0.02039890, 1.78369811, 0.34303909]
    cal_14_mm = [-0.28000178, -0.69325397, -0.03153683, -2.47854166, -0.02039010, 1.77857438, 0.34307510]
    cal_16_mm = [-0.27970461, -0.69325456, -0.03153029, -2.47348600, -0.02037217, 1.77345216, 0.34311173]
    cal_18_mm = [-0.27965020, -0.69325785, -0.03153363, -2.46830834, -0.02035854, 1.76836322, 0.34314925]
    cal_20_mm = [-0.27940375, -0.69325846, -0.03153196, -2.46322564, -0.02033933, 1.76330469, 0.34319068]
    cal_22_mm = [-0.27935815, -0.69324355, -0.03153029, -2.45808868, -0.02032076, 1.75823900, 0.34322271]
    cal_24_mm = [-0.27925438, -0.69322097, -0.03152834, -2.45284836, -0.02030230, 1.75322801, 0.34325413]
    cal_26_mm = [-0.27905731, -0.69318747, -0.03153363, -2.44767785, -0.02028588, 1.74821929, 0.34329462]
    cal_28_mm = [-0.27899975, -0.69285682, -0.03152912, -2.44243555, -0.02026945, 1.74325363, 0.34332533]
    cal_30_mm = [-0.27875733, -0.69262860, -0.03153052, -2.43722129, -0.02024957, 1.73820789, 0.34335799]
    cal_40_mm = [-0.27802718, -0.69101920, -0.03152877, -2.41057300, -0.01999921, 1.71330324, 0.34352739]
    cal_50_mm = [-0.27726699, -0.68847617, -0.03153187, -2.38345928, -0.01994613, 1.68861054, 0.34367666]
    cal_60_mm = [-0.27658418, -0.68519827, -0.03153210, -2.35569011, -0.01986667, 1.66429998, 0.34380664]
    cal_70_mm = [-0.27595735, -0.68126972, -0.03153016, -2.32748400, -0.01970777, 1.63999279, 0.34392638]
    cal_80_mm = [-0.27528157, -0.67652077, -0.03153253, -2.29861205, -0.01958793, 1.61586960, 0.34402378]
  
  elif request.move_to_int == 2:
    # left of diamond
    cal_0_mm = [-0.27616540, 0.15578415, -0.22721627, -1.66631745, 0.03683417, 1.80104259, -0.15528937]
    cal_2_mm = [-0.27617411, 0.15611865, -0.22720995, -1.66253307, 0.03682765, 1.79664971, -0.15528671]
    cal_4_mm = [-0.27620290, 0.15621767, -0.22719797, -1.65822191, 0.03683259, 1.79146907, -0.15528129]
    cal_6_mm = [-0.27623444, 0.15621656, -0.22720170, -1.65326519, 0.03682135, 1.78648827, -0.15526498]
    cal_8_mm = [-0.27629765, 0.15620863, -0.22720381, -1.64836129, 0.03681412, 1.78152018, -0.15522919]
    cal_10_mm = [-0.27657639, 0.15619738, -0.22719451, -1.64350112, 0.03680276, 1.77646662, -0.15517727]
    cal_12_mm = [-0.27661698, 0.15619948, -0.22719995, -1.63854159, 0.03679163, 1.77129102, -0.15512472]
    cal_14_mm = [-0.27670542, 0.15618691, -0.22719704, -1.63342960, 0.03677567, 1.76617633, -0.15507761]
    cal_16_mm = [-0.27693113, 0.15619099, -0.22719776, -1.62817986, 0.03676094, 1.76112790, -0.15503044]
    cal_18_mm = [-0.27698862, 0.15618737, -0.22719823, -1.62305264, 0.03674655, 1.75605491, -0.15499996]
    cal_20_mm = [-0.27727472, 0.15619225, -0.22720042, -1.61800928, 0.03673192, 1.75099851, -0.15496154]
    cal_22_mm = [-0.27732719, 0.15620059, -0.22719776, -1.61285246, 0.03672738, 1.74594592, -0.15493409]
    cal_24_mm = [-0.27761830, 0.15621127, -0.22719776, -1.60745367, 0.03672123, 1.74080161, -0.15490913]
    cal_26_mm = [-0.27768017, 0.15624490, -0.22719609, -1.60202224, 0.03671720, 1.73562052, -0.15489089]
    cal_28_mm = [-0.27797176, 0.15666267, -0.22719609, -1.59660370, 0.03671637, 1.73042549, -0.15487260]
    cal_30_mm = [-0.27804283, 0.15674455, -0.22719442, -1.59125208, 0.03671731, 1.72521208, -0.15486231]
    cal_40_mm = [-0.27908596, 0.15864756, -0.22719971, -1.56324802, 0.03673708, 1.69924396, -0.15485682]
    cal_50_mm = [-0.28017996, 0.16144523, -0.22719442, -1.53414093, 0.03683619, 1.67271702, -0.15486824]
    cal_60_mm = [-0.28142754, 0.16529497, -0.22719080, -1.50345014, 0.03756306, 1.64583815, -0.15494321]
    cal_70_mm = [-0.28313011, 0.17004516, -0.22719442, -1.47119519, 0.03850070, 1.61855458, -0.15532162]
    cal_80_mm = [-0.28489265, 0.17598852, -0.22718913, -1.43758070, 0.03968620, 1.59070500, -0.15582283]
  
  elif request.move_to_int == 3:
    # forward most point (bottom of diamond)
    cal_0_mm = [-0.13584227, 0.42395093, -0.03415631, -1.28013700, 0.01354295, 1.71426353, 0.49537017]
    cal_2_mm = [-0.13587427, 0.42504220, -0.03417534, -1.27507550, 0.01350617, 1.70928657, 0.49536377]
    cal_4_mm = [-0.13594595, 0.42620300, -0.03417701, -1.26920446, 0.01350235, 1.70377379, 0.49536083]
    cal_6_mm = [-0.13611779, 0.42699835, -0.03418294, -1.26282015, 0.01350236, 1.69822350, 0.49536271]
    cal_8_mm = [-0.13614503, 0.42784053, -0.03418307, -1.25653695, 0.01350103, 1.69258609, 0.49536128]
    cal_10_mm = [-0.13616814, 0.42864570, -0.03418149, -1.25016192, 0.01350103, 1.68694952, 0.49535986]
    cal_12_mm = [-0.13618657, 0.42951024, -0.03418126, -1.24356106, 0.01349973, 1.68130257, 0.49536378]
    cal_14_mm = [-0.13620682, 0.43059498, -0.03418507, -1.23679939, 0.01350283, 1.67571533, 0.49536488]
    cal_16_mm = [-0.13624561, 0.43159368, -0.03418169, -1.23011873, 0.01350500, 1.67007934, 0.49536893]
    cal_18_mm = [-0.13651895, 0.43270324, -0.03418827, -1.22336794, 0.01350451, 1.66445276, 0.49536608]
    cal_20_mm = [-0.13653800, 0.43382500, -0.03418660, -1.21634057, 0.01350975, 1.65866646, 0.49536739]
    cal_22_mm = [-0.13656851, 0.43517922, -0.03418827, -1.20920382, 0.01350991, 1.65283949, 0.49536719]
    cal_24_mm = [-0.13659716, 0.43639886, -0.03418652, -1.20225151, 0.01351013, 1.64695829, 0.49537004]
    cal_26_mm = [-0.13662993, 0.43776059, -0.03418827, -1.19500619, 0.01351435, 1.64108506, 0.49537425]
    cal_28_mm = [-0.13689395, 0.43923129, -0.03418827, -1.18767057, 0.01351889, 1.63520519, 0.49537817]
    cal_30_mm = [-0.13692146, 0.44081733, -0.03418827, -1.18011321, 0.01352093, 1.62935390, 0.49538045]
    cal_40_mm = [-0.13737332, 0.44933545, -0.03418994, -1.14119920, 0.01358085, 1.59875841, 0.49539300]
    cal_50_mm = [-0.13798051, 0.45985779, -0.03418994, -1.09851117, 0.01392985, 1.56700412, 0.49541258]
    cal_60_mm = [-0.13872151, 0.47291139, -0.03419356, -1.05193096, 0.01426744, 1.53345538, 0.49541994]
    cal_70_mm = [-0.13953435, 0.48875035, -0.03418994, -1.00029921, 0.01481899, 1.49778465, 0.49543799]
    cal_80_mm = [-0.14047807, 0.50845062, -0.03419356, -0.94229110, 0.01545554, 1.45926596, 0.49545755]
  
  elif request.move_to_int == 4:
    # pre-bin waypoint
    cal_0_mm = [0.46666110, -0.62180026, -0.21519536, -2.45454004, -0.12995350, 1.83887581, 0.20210538]
    cal_2_mm = [0.46717273, -0.62204574, -0.21523148, -2.45091803, -0.12998131, 1.83409420, 0.20216280]
    cal_4_mm = [0.46806175, -0.62201463, -0.21524807, -2.44650184, -0.12995232, 1.82878929, 0.20240184]
    cal_6_mm = [0.46895619, -0.62234723, -0.21524961, -2.44158944, -0.12993926, 1.82363134, 0.20258809]
    cal_8_mm = [0.46975100, -0.62260874, -0.21525128, -2.43681422, -0.12990575, 1.81838208, 0.20277431]
    cal_10_mm = [0.47054904, -0.62296580, -0.21524963, -2.43193338, -0.12986399, 1.81309612, 0.20301506]
    cal_12_mm = [0.47140770, -0.62330342, -0.21525773, -2.42695680, -0.12980299, 1.80787713, 0.20321056]
    cal_14_mm = [0.47232777, -0.62359527, -0.21525805, -2.42215027, -0.12961196, 1.80269351, 0.20341441]
    cal_16_mm = [0.47325566, -0.62373863, -0.21525593, -2.41709049, -0.12953107, 1.79751615, 0.20364414]
    cal_18_mm = [0.47408576, -0.62400285, -0.21526092, -2.41209331, -0.12946550, 1.79237815, 0.20380324]
    cal_20_mm = [0.47485108, -0.62430516, -0.21525816, -2.40710337, -0.12937694, 1.78728708, 0.20402659]
    cal_22_mm = [0.47567336, -0.62435891, -0.21525976, -2.40204178, -0.12928525, 1.78220195, 0.20419636]
    cal_24_mm = [0.47666993, -0.62441036, -0.21525976, -2.39711360, -0.12918041, 1.77704412, 0.20439643]
    cal_26_mm = [0.47754901, -0.62449473, -0.21526145, -2.39198623, -0.12896470, 1.77184067, 0.20458366]
    cal_28_mm = [0.47834403, -0.62472560, -0.21526337, -2.38679634, -0.12890379, 1.76666817, 0.20476005]
    cal_30_mm = [0.47911592, -0.62472554, -0.21526338, -2.38172724, -0.12883189, 1.76160095, 0.20493092]
    cal_40_mm = [0.48334437, -0.62469302, -0.21526671, -2.35566507, -0.12811896, 1.73642197, 0.20575803]
    cal_50_mm = [0.48757992, -0.62330063, -0.21526821, -2.32910876, -0.12743078, 1.71137637, 0.20651196]
    cal_60_mm = [0.49176489, -0.62140499, -0.21526882, -2.30199482, -0.12666095, 1.68640656, 0.20716931]
    cal_70_mm = [0.49580714, -0.61853090, -0.21527244, -2.27424975, -0.12585657, 1.66179994, 0.20769574]
    cal_80_mm = [0.49985603, -0.61493227, -0.21527411, -2.24584691, -0.12499754, 1.63719854, 0.20809502]
    cal_100_mm = [0.50743542, -0.60534878, -0.21527257, -2.18736492, -0.12296777, 1.58859007, 0.20848474]
    cal_120_mm = [0.51457115, -0.59280205, -0.21527619, -2.12590178, -0.12071953, 1.54012589, 0.20846371]
    cal_140_mm = [0.52113863, -0.57739214, -0.21527194, -2.06197085, -0.11826960, 1.49177427, 0.20772678]
    cal_160_mm = [0.52686040, -0.55880475, -0.21527049, -1.99486152, -0.11549934, 1.44342456, 0.20609049]
    cal_180_mm = [0.53161798, -0.53718960, -0.21526671, -1.92471706, -0.11236270, 1.39480298, 0.20364672]
    cal_200_mm = [0.53525643, -0.51237958, -0.21526359, -1.85058538, -0.10876263, 1.34549029, 0.19996557]

  elif request.move_to_int == 5:
    # bin to drop items
    cal_0_mm = [0.58306325, -0.10993592, 0.18835821, -1.99860943, 0.01293729, 1.89987296, 0.77871657]
    cal_2_mm = [0.58289484, -0.11057727, 0.18805577, -1.99476894, 0.01293177, 1.89518433, 0.77869293]
    cal_4_mm = [0.58287315, -0.11046038, 0.18806744, -1.99132001, 0.01297681, 1.89002654, 0.77864572]
    cal_6_mm = [0.58283889, -0.11048578, 0.18806745, -1.98693085, 0.01299048, 1.88494571, 0.77858696]
    cal_8_mm = [0.58251043, -0.11101906, 0.18808230, -1.98240106, 0.01301459, 1.87989716, 0.77838417]
    cal_10_mm = [0.58247333, -0.11149760, 0.18808038, -1.97798252, 0.01306932, 1.87487092, 0.77834512]
    cal_12_mm = [0.58241602, -0.11200933, 0.18807658, -1.97338430, 0.01318510, 1.86989095, 0.77829522]
    cal_14_mm = [0.58215790, -0.11252858, 0.18808399, -1.96865351, 0.01341352, 1.86490773, 0.77821786]
    cal_16_mm = [0.58211709, -0.11300836, 0.18808399, -1.96414034, 0.01343306, 1.85984343, 0.77814958]
    cal_18_mm = [0.58206843, -0.11344786, 0.18808038, -1.95949380, 0.01345161, 1.85477816, 0.77807403]
    cal_20_mm = [0.58177370, -0.11385522, 0.18808232, -1.95481381, 0.01346877, 1.84966578, 0.77800926]
    cal_22_mm = [0.58173095, -0.11418538, 0.18808251, -1.95012062, 0.01348639, 1.84456347, 0.77794463]
    cal_24_mm = [0.58168177, -0.11454292, 0.18808233, -1.94521600, 0.01349741, 1.83949708, 0.77787932]
    cal_26_mm = [0.58138728, -0.11460863, 0.18808400, -1.94049367, 0.01351312, 1.83443650, 0.77779473]
    cal_28_mm = [0.58135018, -0.11503483, 0.18808235, -1.93569329, 0.01352041, 1.82946235, 0.77773587]
    cal_30_mm = [0.58130666, -0.11508727, 0.18808399, -1.93087032, 0.01353576, 1.82447260, 0.77768948]
    cal_40_mm = [0.58066142, -0.11577482, 0.18808232, -1.90602796, 0.01357340, 1.79899213, 0.77743417]
    cal_50_mm = [0.58025236, -0.11579819, 0.18808975, -1.88036124, 0.01357064, 1.77363020, 0.77739584]
    cal_60_mm = [0.57986177, -0.11532467, 0.18808711, -1.85380943, 0.01353948, 1.74813091, 0.77738918]
    cal_70_mm = [0.57959856, -0.11364394, 0.18808552, -1.82633181, 0.01343700, 1.72238786, 0.77739669]
    cal_80_mm = [0.57934965, -0.11087069, 0.18808268, -1.79790338, 0.01293385, 1.69660742, 0.77745827]
    cal_100_mm = [0.57932559, -0.10302034, 0.18807857, -1.73794573, 0.01152313, 1.64433749, 0.77832535]
    cal_120_mm = [0.57962400, -0.09170179, 0.18807330, -1.67345904, 0.00934031, 1.59122620, 0.77961864]
    cal_140_mm = [0.58078174, -0.07612561, 0.18807117, -1.60377829, 0.00651070, 1.53672905, 0.78142983]
    cal_160_mm = [0.58309017, -0.05653660, 0.18805531, -1.52843322, 0.00274478, 1.48057941, 0.78395259]
    cal_180_mm = [0.58659586, -0.03203691, 0.18804458, -1.44579159, -0.00192619, 1.42212148, 0.78709276]
    cal_200_mm = [0.59167338, -0.00159176, 0.18803158, -1.35439760, -0.00779925, 1.36037718, 0.79090279]

  else:
    rospy.logerr(f"move_panda_to() received unknown integer = {request.move_to_int}")
    return False
  
  allow_noise = False
  
  calibrated_0mm = 4# + umh # height in mm where table touch occurs with above calibration

  # beyond this we do not have additional 2mm increments
  if request.height_mm >= 28: calibrated_0mm = 0

  # calculate the desired reset height
  if request is None:
    reset_to = 10 # default, panda is limited to +-30mm in move_panda_z_abs
  else:
    reset_to = request.height_mm
  reset_to += calibrated_0mm
  if panda_reset_noise_mm is not None and allow_noise:
    reset_to += (random.random() * 2 * panda_reset_noise_mm) - panda_reset_noise_mm

  target_state_name = height_to_cal(reset_to)
  rospy.loginfo(f"move_panda_to() will reset to a height given by '{target_state_name}'")

  # assign this reset height to a state vector from above (eg 'cal_10_mm')
  try:
    target_state = eval(f"{target_state_name}")
  except NameError as e:
    rospy.logerr(f"move_panda_to() error: desired target height has no calibrated value")
    raise RuntimeError(f"move_panda_to() error: {e}")

  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.15 # 0.1 is slow movements

  if use_panda_threads:
    process = Process(target=franka_instance.move_joints, args=(target_state, speed_factor))
    process.start()
    process.join()
    process.close()
    if process.exception:
      rospy.logwarn(f"Panda movement exception, likely too high force on the table. Cancelling grasping. Error message: {process.exception}")
      cancel_grasping_callback()
  else:
    franka_instance.move_joints(target_state, speed_factor) # now approach reset height

  panda_z_height = 0

  if log_level > 0: 
    rospy.loginfo(f"panda reset to position {request.move_to_int} and a calibrated height of {2 * round(reset_to / 2) - calibrated_0mm} (rounded from {reset_to:.1f} and calibrated zero of {calibrated_0mm})")

  return True


  #--------------------------------#
  
  if abs(request.height_mm - 10) < 1e-3:
    target_state = cal_12_mm # choose this height for good behaviour 10/12/14
  elif abs(request.height_mm - 20) < 1e-3:
    target_state = cal_50_mm
  elif abs(request.height_mm - 30) < 1e-3:
    target_state = cal_50_mm
  elif abs(request.height_mm - 50) < 1e-3:
    target_state = cal_50_mm
  else:
    rospy.logerr(f"move_panda_to() received unknown height_mm = {request.height_mm}")
    return False
  
  # move the joints slowly to the reset position - this could be dangerous!
  speed_factor = 0.05 # 0.1 is slow movements, 0.05 very slow, 0.01 extremely slow

  if use_panda_threads:
    process = Process(target=franka_instance.move_joints, args=(target_state, speed_factor))
    process.start()
    process.join()
    process.close()
    if process.exception:
      rospy.logwarn(f"Panda movement exception, likely too high force on the table. Cancelling grasping. Error message: {process.exception}")
      cancel_grasping_callback()
  else:
    franka_instance.move_joints(target_state, speed_factor) # now approach reset height

  return True

def demo_movement(goto, pregrasp=False, grasp=False, place=False):
  """
  Move from current point to the next point
  """

  global continue_demo, continue_grasping, move_panda, demo_panda_is_at

  move_msg = PandaMoveToInt()
  continue_demo = True

  if goto not in [0, 1, 2, 3, 4, 5]:
    rospy.logerr(f"demo_movement() given invalid goto = {goto}")
    return False
  
  grasp_height = 80
  bin_height = 200
  
  # grasping positions
  if goto in [0, 1, 2, 3]:
    height = grasp_height

  # bin placement positions
  elif goto in [4, 5]:
    height = bin_height

  # go to the specified point
  move_msg.move_to_int = goto
  move_msg.height_mm = height
  moved = move_panda_to(move_msg)

  if not moved:
    rospy.logerr("Panda motion error in demo_movement(), aborting all")
    return False
  elif not continue_demo:
    rospy.loginfo("Demo cancelled, demo_movement() is returning")
    return True
  elif rospy.is_shutdown(): return False

  # record that the panda is now at the new location
  demo_panda_is_at = goto

  if pregrasp or grasp:

    # lower to grasping height and reset everything ready
    rospy.sleep(0.3)
    move_msg.move_to_int = goto
    move_msg.height_mm = 10
    moved = move_panda_to(move_msg)

    if not moved:
      rospy.logerr("Panda motion error in demo_movement(), aborting all")
      return False
    elif not continue_demo:
      rospy.loginfo("Demo cancelled, demo_movement() is returning")
      return True
    elif rospy.is_shutdown(): return False

    # reset
    reset_all(skip_panda=True)
    global panda_z_height
    panda_z_height = 0

    # wait for gripper to finish
    time.sleep(1)
    while not rospy.is_shutdown():
      if not gripper_target_reached:
        time.sleep(0.05)
      else: break

    if grasp:
    
      # do a grasp
      execute_grasping_callback(reset=False)

      # wait for the grasp to execute or be stopped
      rospy.sleep(0.3)
      while continue_grasping:
        rospy.sleep(0.1)
        if not continue_demo or not continue_grasping:
          rospy.loginfo("Demo cancelled, demo_movement() is returning")
          return True
        elif rospy.is_shutdown(): return False

      # when grasp is done, lift to 50mm
      move_msg.move_to_int = goto
      move_msg.height_mm = grasp_height
      moved = move_panda_to(move_msg)

      if not moved:
        rospy.logerr("Panda motion error in demo_movement(), aborting all")
        return False
      elif not continue_demo:
        rospy.loginfo("Demo cancelled, demo_movement() is returning")
        return True
      elif rospy.is_shutdown(): return False

  elif place:

    # # lower before placing
    # move_msg.move_to_int = goto
    # move_msg.height_mm = 20
    # moved = move_panda_to(move_msg)

    gentle_place()

    # # when place is done, lift to 50mm
    # move_msg.move_to_int = goto
    # move_msg.height_mm = 50
    # moved = move_panda_to(move_msg)

    if not moved:
      rospy.logerr("Panda motion error in demo_movement(), aborting all")
      return False
    elif not continue_demo:
      rospy.loginfo("Demo cancelled, demo_movement() is returning")
      return True
    elif rospy.is_shutdown(): return False

  return True

def loop_demo(request=None):
  """
  Run the demo in a continous loop
  """

  global demo_loop_int, continue_demo
  continue_demo = True

  # apply any step skips (forward or backwards)
  if request is not None:
    demo_loop_int += request.skip

  # loop_to_do = [
  #   ["grasp", 1],
  #   ["place", 2],
  #   ["grasp", 4],
  #   ["place", 1],
  #   ["grasp", 3],
  #   ["place", 4],
  #   ["grasp", 2],
  #   ["place", 3],
  # ]

  loop_to_do = [
    ["grasp", 0],
    ["move", 4],
    ["place", 5],
    ["move", 4],
    ["grasp", 1],
    ["move", 4],
    ["place", 5],
    ["move", 4],
    ["grasp", 2],
    ["move", 4],
    ["place", 5],
    ["move", 4],
    ["grasp", 3],
    ["move", 4],
    ["place", 5],
    ["move", 4],
  ]

  actually_grasp = True

  while not rospy.is_shutdown() and continue_demo:

    (command, location) = loop_to_do[demo_loop_int]

    if command == "grasp":
      if actually_grasp:
        success = demo_movement(goto=location, grasp=True)
      else:
        success = demo_movement(goto=location, pregrasp=True)

    elif command == "place":
      success = demo_movement(goto=location, place=True)

    elif command == "move":
      success = demo_movement(goto=location)

    if not success:
      rospy.logerr("Failed movement in loop_demo(), aborting all")
      return False
    
    demo_loop_int += 1

    if demo_loop_int >= len(loop_to_do):
      demo_loop_int = 0

    time.sleep(0.1) # to prevent too many panda motions
    
  return True

def start_demo(request=None):
  """
  Run the demo, use stop_demo to cancel
  """

  global continue_demo, continue_grasping, move_panda, demo_panda_is_at

  # reset to location 1, height 50mm
  good_reset = reset_demo()
  if not good_reset:
    rospy.logerr("Failed reset in start_demo(), aborting all")
    return False

  global demo_loop_int
  demo_loop_int = 0

  loop_demo()

  return []

def stop_demo(request=None):
  """
  Stop the demo
  """

  global continue_demo, continue_grasping
  continue_demo = False
  
  if continue_grasping:
    cancel_grasping_callback()

  return []

def reset_demo(request=None):
  """
  Reset the demo
  """

  global continue_demo, continue_grasping, demo_panda_is_at, move_panda

  # must be 0 (centre), 1 (top left), 2 (bottom left), 3 (bottom right), 4 (top right)
  reset_position = 0

  # end grasping and open the gripper
  if continue_demo or continue_grasping:
    stop_demo()

  # full reset of the gripper, but excluding the panda
  reset_all(skip_panda=True)

  move_msg = PandaMoveToInt()

  if demo_panda_is_at in [0, 1, 2, 3, 4, 5]:

    if demo_panda_is_at in [0, 1, 2, 3]:

      # lift the panda first
      move_msg.move_to_int = demo_panda_is_at
      move_msg.height_mm = 80
      moved = move_panda_to(move_msg)

    elif demo_panda_is_at in [4, 5]:

      # go to the bin waypoint
      move_msg.move_to_int = 4
      move_msg.height_mm = 200
      moved = move_panda_to(move_msg)

    if not moved:
      rospy.logerr("Panda motion error in reset_demo(), aborting all")
      return False

  # now reset to the start position
  if reset_position not in [0, 1, 2, 3, 4, 5]: reset_position = 0

  if demo_panda_is_at != reset_position:
    move_msg.move_to_int = reset_position
    move_msg.height_mm = 80
    moved = move_panda_to(move_msg)

  if not moved:
    rospy.logerr("Panda motion error in reset_demo(), aborting all")
    return False
  
  return True

def print_panda_state(request=None):
  """
  Print the panda state
  """

  state = franka_instance.getState()

  vec_str = "[{0:.8f}, {1:.8f}, {2:.8f}, {3:.8f}, {4:.8f}, {5:.8f}, {6:.8f}]".format(*state.q)

  print("Panda joint state:")
  print(vec_str)

  return []

hardcoded_object_force_measuring_green = {
  "2" : {
    "X" : [-0.28069337, -0.18465624, 0.11252855, -1.79308801, 0.00557113, 1.63109549, 0.22820591],
    "Y" : [-0.28067816, -0.20041854, 0.12062921, -1.81007805, 0.00557847, 1.63082551, -0.18339117],
  },
  "4" : {
    "X" : [-0.28060156, -0.23367603, 0.16820210, -1.85873665, 0.00614091, 1.69284216, 0.51941030],
    "Y" : [-0.28038861, -0.21276796, 0.16017152, -1.86001793, 0.00612240, 1.68230830, 0.08397833],
  },
  "6" : {
    "X" : [-0.27871970, -0.11468955, 0.23129969, -1.77778824, 0.00761231, 1.66875519, 0.76874122],
    "Y" : [-0.28046264, -0.10543998, 0.23374335, -1.76794357, 0.00584989, 1.65160690, 0.07113314],
  },
  "8" : {
    "X" : [-0.28091029, -0.16420879, 0.15070010, -1.88149919, 0.00120455, 1.73359205, 0.44847199],
    "Y" : [-0.28072572, -0.15555267, 0.15084620, -1.88166048, 0.00122767, 1.73450316, 0.02170936],
  },
  "10" : {
    "X" : [-0.27469432, -0.18660782, 0.17350099, -1.79283709, 0.00682472, 1.64312909, 0.49513677],
    "Y" : [-0.27242907, -0.17497830, 0.18407547, -1.80016360, 0.00494904, 1.64173521, 0.04556134],
  },
  "12" : {
    "X" : [-0.27469746, -0.17492785, 0.21402303, -1.75574718, 0.00691229, 1.57555222, 0.50163380], #[-0.28056639, -0.06934160, 0.18793227, -1.66450525, 0.00657487, 1.61252933, 0.90604727],
    "Y" : [-0.18743505, -0.09945260, 0.35270521, -1.67386494, 0.00567364, 1.58914825, -0.31718803], #[-0.19389855, -0.06905497, 0.34948230, -1.67012694, 0.00364755, 1.62753837, -0.31871014],
  },
  "14" : {
    "X" : [-0.27606672, -0.13369937, 0.21072207, -1.79500398, 0.00510505, 1.69592195, 0.81058123],
    "Y" : [-0.25066182, -0.13217422, 0.24753858, -1.74344409, -0.00369604, 1.61144976, -0.43415420],
  },
  "16" : {
    "X" : [-0.27643267, -0.13256331, 0.13743127, -1.70460825, 0.00226769, 1.55355555, 0.28268021],
    "Y" : [-0.26965222, -0.13548586, 0.14517343, -1.70662584, 0.00205102, 1.56661795, -0.14188716],
  },
  "18" : {
    "X" : [-0.26172567, -0.06315081, 0.20643580, -1.64939824, -0.04018200, 1.58129323, 1.04921773],
    "Y" : [-0.29219042, -0.17514905, 0.21090864, -1.77085120, 0.00254249, 1.63615826, 0.69604525],
  },
  "20" : {
    "X" : [-0.27637320, -0.22302036, 0.25724313, -1.86360882, 0.02409553, 1.64668284, 1.01960013],
    "Y" : [-0.28444011, -0.11949539, 0.23183313, -1.77593529, 0.01693870, 1.67218893, 0.87288591],
  },
  "22" : {
    "X" : [-0.27534684, -0.15339331, 0.16271155, -1.80162031, 0.01187669, 1.66734649, 0.61472686],
    "Y" : [-0.26185587, -0.16152694, 0.19397875, -1.80414769, -0.00820810, 1.64714971, -0.00852966],
  },
  "24" : {
    "X" : [-0.27188509, -0.06363571, 0.23049632, -1.64849280, -0.01652225, 1.48626309, 1.03559028],
    "Y" : [-0.17227974, -0.10750029, 0.34341929, -1.72045947, -0.00156884, 1.66381998, -0.24240502],
  },
  "26" : {
    "X" : [-0.27697475, -0.14601591, 0.22186620, -1.77951218, 0.00268244, 1.66197248, 0.73880315],
    "Y" : [-0.27499572, -0.15417218, 0.22693995, -1.78087218, 0.00223199, 1.64768070, 0.15117543],
  },
  "28" : {
    "X" : [-0.27611278, -0.19119269, 0.16501353, -1.76116749, 0.02709927, 1.61511364, 0.58347988],
    "Y" : [-0.27799380, -0.21297657, 0.15657929, -1.76939147, 0.02562485, 1.60693449, 0.01892179],
  },
  "30" : {
    "X" : [-0.28173900, -0.11908118, 0.26559166, -1.81942041, -0.03252781, 1.72787966, 0.90076040],
    "Y" : [-0.26307180, -0.12834553, 0.26490279, -1.82014274, -0.03056514, 1.72639261, 0.32351686],
  },
}

hardcoded_object_force_measuring_ycb = {
  "1" : {
    "X" : [-0.36822848, -0.24370299, 0.26221085, -1.82266134, 0.00660396, 1.63084830, 0.04742342],
    "Y" : [-0.34878434, -0.19379540, 0.22452411, -1.77025586, 0.00845984, 1.60588533, 0.03239621],
  },
  "2" : {
    "X" : [-0.38938788, -0.21589496, 0.25725081, -1.79024386, 0.01695426, 1.61316229, 0.23300391],
    "Y" : [-0.34964094, -0.25129792, 0.20835552, -1.83186395, 0.00998687, 1.63875029, -0.06993842],
  },
  "3" : {
    "X" : [-0.39849714, -0.17619944, 0.25171656, -1.74413389, 0.01700903, 1.58506178, 0.34202753],
    "Y" : [-0.35814276, -0.19549277, 0.23708305, -1.76393569, 0.00889229, 1.59696620, -0.05891695],
  },
  "4" : {
    "X" : [-0.42908830, -0.25237199, 0.29438708, -1.83039462, 0.00104007, 1.60534001, 1.36668947],
    "Y" : [-0.40696170, -0.20175402, 0.27362313, -1.76671062, -0.00176479, 1.59019100, -0.14504454],
  },
  "5" : {
    "X" : [-0.33734248, -0.22562461, 0.21329443, -1.81359614, -0.00115945, 1.62097849, 0.27809521],
    "Y" : [-0.31184307, -0.22979622, 0.16276000, -1.83101824, 0.01560823, 1.62756959, -0.13225184],
  },
  "6" : {
    "X" : [-0.35113246, -0.21038871, 0.20779082, -1.79192823, 0.01264486, 1.61473502, 0.50283690],
    "Y" : [-0.37982736, -0.22834180, 0.22571717, -1.80426171, 0.02785046, 1.61833920, 0.00403935],
  },
  "7" : {
    "X" : [-0.46441629, -0.23208175, 0.31465964, -1.80422963, 0.03132587, 1.61427766, -0.32375088],
    "Y" : [-0.28523755, -0.12735198, 0.31327087, -1.66000702, -0.03820321, 1.57394596, 0.26259663],
  },
  "8" : {
    "X" : [-0.49468934, -0.33287497, 0.36455098, -1.91088287, 0.06234966, 1.68656495, -0.14729543],
    "Y" : [-0.31960041, -0.21811382, 0.38460827, -1.76505973, 0.05219151, 1.59815581, 1.62348173],
  },
  "9" : {
    "X" : [-0.35091209, -0.15675582, 0.22496334, -1.72634900, 0.03306611, 1.58245729, 0.62402635],
    "Y" : [-0.34559731, -0.24551939, 0.25637673, -1.85514882, 0.00940002, 1.64749501, -0.00985726],
  },
  "10" : {
    "X" : [-0.33302105, -0.23920591, 0.20862076, -1.82806662, 0.03724969, 1.64035617, -0.11816009],
    "Y" : [-0.35600591, -0.16159757, 0.25744177, -1.81287052, 0.00821057, 1.68553400, 0.76692049],
  },
  "11" : {
    "X" : [-0.30908622, -0.21088443, 0.17168812, -1.78770874, 0.04277118, 1.62067892, 0.55839070],
    "Y" : [-0.32503490, -0.15846392, 0.18842926, -1.69382091, 0.03550327, 1.54220376, -0.10574355],
  },
  "12" : {
    "X" : [-0.29576385, -0.16584597, 0.28066389, -1.79689273, 0.02170473, 1.63896001, 0.57205689],
    "Y" : [-0.36313028, -0.07106410, 0.26736304, -1.65453560, 0.05581303, 1.60219679, 0.98449366],
  },
  "13" : {
    "X" : [-0.38268714, -0.17334462, 0.25190128, -1.75094423, 0.02115341, 1.60080303, -0.10838403],
    "Y" : [-0.36415470, -0.11309172, 0.26738716, -1.61339559, 0.01974911, 1.52049612, 0.82221509],
  },
  "14" : {
    "X" : [-0.34708535, -0.24665460, 0.22070357, -1.85637020, 0.06002641, 1.62860031, 1.11184908],
    "Y" : [-0.31142471, -0.21162924, 0.14801621, -1.82123772, 0.04136386, 1.65757537, 0.62377793],
  },
  "15" : {
    "X" : [-0.33771638, -0.21299671, 0.23698876, -1.82735471, -0.02304901, 1.63979386, -0.11648936],
    "Y" : [-0.31621413, -0.25791378, 0.24425740, -1.83870807, 0.01254511, 1.59059062, 0.99776312],
  },
  "16" : {
    "X" : [-0.33679335, -0.34140686, 0.22445922, -2.03388591, -0.00107697, 1.77158404, -0.36154753],
    "Y" : [-0.33196573, -0.27557177, 0.23241505, -1.96176534, 0.01486781, 1.71198869, 1.22303114],
  },
  "17" : {
    "X" : [-0.27970909, -0.22227602, 0.20857714, -1.80011564, 0.01011829, 1.60661487, -0.03122316],
    "Y" : [-0.32448044, -0.17954766, 0.20391652, -1.78247072, 0.01073538, 1.65693519, 0.74255822],
  },
  "18" : {
    "X" : [-0.32820666, -0.17805193, 0.19110956, -1.75821472, 0.00861790, 1.60394009, 0.54447852],
    "Y" : [-0.28030205, -0.25169401, 0.17057721, -1.83578748, 0.00854337, 1.63339174, -0.10541317],
  },
}

# ----- scripting to initialise and run node ----- #

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("rl_grasping_node")
  rospy.loginfo("rl_grasping_node has now started")
  
  # what namespace will we use in this node for publishers/services
  node_ns = "rl" # gripper/rl

  # get input parameters - are we allowing robot movement?
  move_gripper = True # rospy.get_param(f"/{node_ns}/move_gripper")
  move_panda = True # rospy.get_param(f"/{node_ns}/dqn/move_panda")
  rospy.loginfo(f"  > move gripper is {move_gripper}")
  rospy.loginfo(f"  > move panda is {move_panda}")
  rospy.loginfo(f"  > using camera is {camera}")
  rospy.loginfo(f"  > using devel model is {use_devel}")
  rospy.loginfo(f"  > photoshoot calibration is {photoshoot_calibration}")
  rospy.loginfo(f"  > use panda threads is {use_panda_threads}")
  rospy.loginfo(f"  > reject wrist noise is {reject_wrist_noise}")
  rospy.loginfo(f"  > prevent table hit is {prevent_table_hit}")

  # do we need to import the franka control library
  if move_panda: connect_panda()
    
  # subscriber for gripper data and publisher to send gripper commands
  rospy.Subscriber("/gripper/real/state", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  if extra_gripper_measuring:
    rospy.Subscriber("/gripper/other/output", GripperOutput, extra_gripper_data_callback)
    extra_demand_pub = rospy.Publisher("/gripper/other/request", GripperRequest)
    extra_zfrc_pub = rospy.Publisher(f"/{node_ns}/other_frc", Float32)

  # subscribers for image topics
  rospy.Subscriber("/camera/rgb", Image, rgb_callback)
  rospy.Subscriber("/camera/depth", Image, depth_callback)

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher(f"/{node_ns}/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher(f"/{node_ns}/sensor", NormalisedSensor, queue_size=10)
  palm_SI_pub = rospy.Publisher(f"/{node_ns}/palm_SI", Float32, queue_size=10)
  debug_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)
  panda_local_wrench_pub = rospy.Publisher(f"/{node_ns}/panda/local_wrench", Wrench, queue_size=10)
  panda_global_wrench_pub = rospy.Publisher(f"/{node_ns}/panda/global_wrench", Wrench, queue_size=10)
  img_obs_pub = rospy.Publisher(f"/{node_ns}/img_obs", Image, queue_size=10)

  # publish the first message to rqt keeps things going
  norm_state_pub.publish(NormalisedState())
  norm_sensor_pub.publish(NormalisedSensor())

  # user set - what do we load by default
  if use_devel and False:

    rospy.logwarn("Loading DEVEL policy")

    # load a devel training
    load = LoadModel()
    load.folderpath = "/home/luke/mujoco-devel/rl/models"
    load.run_id = "best"

    # # Program: paper_baseline_1_extended
    # load.timestamp =  "08-12-23_19-19"
    # load.job_number = 53
    # load.run_id = "best"

    # # Program: continue_good_curriculum
    # load.timestamp =  "05-02-24_11-37"
    # load.job_number = 4
    # load.run_id = "best"

    # Program: paper_baseline_1_rigid_fingers
    load.timestamp =  "07-02-24_18-22"
    load.job_number = 4
    load.run_id = "best"

    # Program: palm_vs_no_palm_1
    load.timestamp = "19-01-24_16-54"
    load.run_id = "best"
    # load.job_number = 6 # no palm, 45deg, E1
    # load.job_number = 16 # no palm, 60deg, E1
    # load.job_number = 28 # no palm, 75deg, E1
    # load.job_number = 37 # no palm, 90deg, E1
    # load.job_number = 44 # palm, 45deg, E1
    # load.job_number = 53 # palm, 60deg, E1
    load.job_number = 61 # palm, 75deg, E1
    # load.job_number = 78 # palm, 90deg, E1

    load_model(load)

    # # special case for testing GAN image rendering
    # model.trainer.env.set_device("cuda")
    # model.trainer.env.params.use_rgb_rendering = True
    # model.trainer.env.params.rgb_rendering_method = "cyclegan_encoder"
    # model.trainer.env._load_image_rendering_model(device="cuda", loadA=False) # loadB for reverse transform
    # model.trainer.agent.set_device(model.trainer.env.torch_device)
    # print("READY TO RENDER")

  elif use_devel:

    # load a specific model baseline
    load = LoadBaselineModel()
    load.thickness = 0.9e-3
    load.width = 28e-3
    load.sensors = 3
    load_baseline_1_model(load)

    # make forwards compatible
    model.trainer.env.params.use_rgb_in_observation = False

  else:

    # load a specific model baseline
    load = LoadBaselineModel()
    load.thickness = 1.0e-3
    load.width = 24e-3
    load.sensors = 3
    load_baseline_1_model(load)

  if log_level >= 3:
    # turn on all debug information
    model.trainer.env.log_level = 2
    model.trainer.env.mj.set.debug = True

  # ensure the model is in testing mode (disables any action noise/dropout etc)
  model.trainer.agent.testing_mode()

  # begin services for this node
  rospy.loginfo(f"rl_grasping_node services now available under namespace /{node_ns}/")
  rospy.Service(f"/{node_ns}/start", Empty, execute_grasping_callback)
  rospy.Service(f"/{node_ns}/stop", Empty, cancel_grasping_callback)
  rospy.Service(f"/{node_ns}/reset", Empty, reset_all)
  rospy.Service(f"/{node_ns}/reset_panda", ResetPanda, reset_panda)
  rospy.Service(f"/{node_ns}/reset_gripper", Empty, reset_gripper)
  rospy.Service(f"/{node_ns}/move_panda", PandaMoveZ, move_panda_service)
  rospy.Service(f"/{node_ns}/connect_panda", Empty, connect_panda)
  rospy.Service(f"/{node_ns}/load_model", LoadModel, load_model)
  rospy.Service(f"/{node_ns}/apply_settings", ApplySettings, apply_settings)
  rospy.Service(f"/{node_ns}/debug_gripper", Empty, debug_gripper)
  rospy.Service(f"/{node_ns}/test", StartTest, start_test)
  rospy.Service(f"/{node_ns}/heuristic_test", StartTest, heuristic_test)
  rospy.Service(f"/{node_ns}/trial", StartTrial, start_trial)
  rospy.Service(f"/{node_ns}/save_test", Empty, save_test)
  rospy.Service(f"/{node_ns}/end_test", Empty, end_test)
  rospy.Service(f"/{node_ns}/save_trial", SaveTrial, save_trial)
  rospy.Service(f"/{node_ns}/delete_trial", Empty, delete_last_trial)
  rospy.Service(f"/{node_ns}/print_test", Empty, print_test_results)
  rospy.Service(f"/{node_ns}/load_baseline_model", LoadBaselineModel, load_baseline_1_model)
  rospy.Service(f"/{node_ns}/set_use_sim_ft_sensor", SetBool, set_sim_ft_sensor_callback)
  rospy.Service(f"/{node_ns}/set_dynamic_recal_ft", SetBool, set_dynamic_recal_ft_callback)
  rospy.Service(f"/{node_ns}/make_test_heuristic", Empty, make_test_heurisitc)
  rospy.Service(f"/{node_ns}/place", Empty, gentle_place)
  rospy.Service(f"/{node_ns}/panda_move_to_int", PandaMoveToInt, move_panda_to)
  rospy.Service(f"/{node_ns}/start_demo", Empty, start_demo)
  rospy.Service(f"/{node_ns}/stop_demo", Empty, stop_demo)
  rospy.Service(f"/{node_ns}/reset_demo", Empty, reset_demo)
  rospy.Service(f"/{node_ns}/continue_demo", Demo, loop_demo)
  rospy.Service(f"/{node_ns}/set_reject_wrist_noise", SetBool, set_reject_wrist_noise_callback)
  rospy.Service(f"/{node_ns}/extra_frc_test", ForceTest, run_extra_force_test)
  rospy.Service(f"/{node_ns}/print_panda_state", Empty, print_panda_state)
  rospy.Service(f"/{node_ns}/stop_force_test", Empty, stop_force_test)

  try:
    while not rospy.is_shutdown(): rospy.spin() # and wait for service requests
  except Exception as e:
    end_test()
    rospy.logerror(f"dqn_node(...) failed, saved test data - exception is {e}")
  