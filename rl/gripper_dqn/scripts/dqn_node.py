#!/home/luke/pyenv/py38_ros/bin/python

import rospy
import sys
import os
import numpy as np
from datetime import datetime
import multiprocessing as mp

# from: https://stackoverflow.com/questions/19924104/python-multiprocessing-handling-child-errors-in-parent
class Process(mp.Process):
  def __init__(self, *args, **kwargs):
    mp.Process.__init__(self, *args, **kwargs)
    self._pconn, self._cconn = mp.Pipe()
    self._exception = None

  def run(self):
    try:
      mp.Process.run(self)
      self._cconn.send(None)
    except Exception as e:
      # tb = traceback.format_exc()
      # self._cconn.send((e, tb))
      raise e  # You can still rise this exception if you need to

  @property
  def exception(self):
    if self._pconn.poll():
      self._exception = self._pconn.recv()
    return self._exception

from std_srvs.srv import Empty, SetBool
from gripper_msgs.msg import GripperState, GripperDemand, GripperInput
from gripper_msgs.msg import NormalisedState, NormalisedSensor
from gripper_dqn.srv import LoadModel, ApplySettings, ResetPanda
from gripper_dqn.srv import StartTest, StartTrial, SaveTrial, LoadBaselineModel
from gripper_dqn.srv import PandaMoveToInt, Demo

from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

# import the test data structure class
from grasp_test_data import GraspTestData

# ----- essential settings and global variable declarations ----- #

# important user settings
camera = False                  # do we want to take camera images, is the camera connected
use_devel = False               # do we load trainings from mujoco-devel and run with that compilation
photoshoot_calibration = False  # do we grasp in ideal position for taking side-on-videos
use_panda_threads = False       # do we run panda in a seperate thread
log_level = 2                   # node log level, 0=disabled, 1=essential, 2=debug, 3=all
action_delay = 0.1              # safety delay between action generation and publishing
panda_z_move_duration = 0.5     # how long to move the panda z height vertically up
panda_reset_height_mm = 10      # real world panda height to reset to before a grasp
scale_gauge_data = 1.0          # scale real gauge data by this
scale_wrist_data = 1.0          # scale real wrist data by this
scale_palm_data = 1.0           # scale real palm data by this
image_rate = 1                  # 1=take pictures every step, 2=every two steps etc
image_batch_size = 1            # 1=save pictures every trial, 2=every two trials etc

# important paths
test_save_path = "/home/luke/gripper-ros/"
if use_devel: mymujoco_rl_path = "/home/luke/mujoco-devel/rl"
else: mymujoco_rl_path = "/home/luke/mymujoco/rl"
# pyfranka_path = "/home/luke/franka/franka_interface/build"
pyfranka_path = "/home/luke/libs/franka/franka_interface/build"

# experimental feature settings
use_sim_ftsensor = False        # use a simulated ft sensor instead of real life
dynamic_recal_ftsensor = False  # recalibrate ftsensor to zero whenever connection is lost
# sim_ft_sensor_step_offset = 2   # lower simulated ft sensor gripper down X steps
# prevent_back_palm = False       # swap any backward palm actions for forwards
prevent_x_open = False           # swap action 1 (X open) for action 2 (Y close)
render_sim_view = False         # render simulated gripper (CRASHES ON 2nd GRASP)
quit_on_palm = None               # quit grasping with a palm value above this (during test only), set None for off
reject_wrist_noise = False       # try to ignore large spikes in the wrist sensor
prevent_table_hit = True         # try to avoid actions which could hit the table

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
demo_loop_int = -1              # where in the demo loop are we
risk_table_hit = False

# declare global variables, these will be overwritten
step_num = 0                    # number of steps in our grasp
panda_z_height = 0.0            # panda z height state reading (=0 @ 10mm above gnd)
current_test_data = None        # current live test data structure
ftenv = None                    # simulated environment for fake force/torque sensor readings
norm_state_pub = None           # ROS publisher for normalised state readings
norm_sensor_pub = None          # ROS publisher for normalised sensor readings
debug_pub = None                # ROS publisher for gripper debug information
last_ft_value = None            # last ftsensor value, used to detect lost connection

# dummy define camera function for cases where no camera is loaded
depth_camera_connected = False
def get_depth_image():
  """
  Empty function (returning rgb, depth) since no camera connected
  """
  rospy.logwarn("Warning: no camera connected, no images collected")
  return None, None

# try to start the camera connection and override the camera function
try:
  if camera is False: raise NotImplementedError
  depth_camera_connected = True
  from capture_depth_image import get_depth_image
except NotImplementedError: rospy.loginfo("use camera is False, no images will be taken")
except Exception as e: 
  rospy.logerr("DEPTH CAMERA NOT CONNECTED but camera is True, beware")
  rospy.logerr(f"Depth camera error message: {e}")

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
  global last_ft_value
  if dynamic_recal_ftsensor:
    # rospy.loginfo(f"last value = {last_ft_value}, new value = {state.ftdata.force.z}")
    # if new value and old are exactly floating-point equal
    if state.ftdata.force.z == last_ft_value:
      model.env.mj.real_sensors.wrist_Z.offset = last_ft_value
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

  # if we are trying to prevent hitting the table
  if prevent_table_hit:
    global risk_table_hit
    if norm_sensor.wrist_z > 0.99 and norm_sensor.palm > 0.99:
      risk_table_hit = True
    else:
      risk_table_hit = False

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

  # prevent the controller from hitting the table
  if prevent_table_hit and action == 6 and risk_table_hit:
    action = 7
    rospy.loginfo("Risk of table hit")

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

  if use_panda_threads:
    process = Process(target=franka.move, args=("relative", T, panda_z_move_duration))
    process.start()
    process.join()
  else:
    franka.move("relative", T, panda_z_move_duration)

  # update with the new target position
  panda_z_height = target_z

  return

def execute_grasping_callback(request=None, reset=True):
  """
  Service callback to complete a dqn grasping task
  """

  if log_level > 0: rospy.loginfo("dqn node is now starting a grasping task")

  global ready_for_new_action
  global action_delay
  global continue_grasping
  global panda_z_height
  global step_num

  if reset:
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

def reset_all(request=None, skip_panda=False):
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
  if move_panda and not skip_panda:
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
  global photoshoot_calibration

  if not move_panda:
    if log_level > 0: rospy.logwarn("asked to reset_panda() but move_panda is false")
    return False

  if request is None:
    reset_to = 10 # default, panda is limited to +-30mm in move_panda_z_abs
  else:
    reset_to = request.reset_height_mm

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

    # new calibrations 23/8/23 after robby, 4=initial touch, 0=v. firm, -2=fails
    cal_0_mm = [-0.10035520, 0.13415142, 0.05034569, -1.54992771, -0.02469004, 1.72361859, 0.82542563]
    cal_2_mm = [-0.10056089, 0.13577175, 0.05015793, -1.54504811, -0.02486288, 1.71835949, 0.82536323]
    cal_4_mm = [-0.10035362, 0.13717769, 0.05021119, -1.53977199, -0.02482598, 1.71273661, 0.82528626]
    cal_6_mm = [-0.10033584, 0.13799006, 0.05020785, -1.53366338, -0.02482757, 1.70740294, 0.82521534]
    cal_8_mm = [-0.10030479, 0.13879796, 0.05020785, -1.52739544, -0.02483351, 1.70205378, 0.82514384]
    cal_10_mm = [-0.10026742, 0.13964332, 0.05020785, -1.52111299, -0.02483833, 1.69663203, 0.82509060]
    cal_12_mm = [-0.10010887, 0.14053731, 0.05020618, -1.51491904, -0.02485414, 1.69118643, 0.82503248]
    cal_14_mm = [-0.10008372, 0.14145294, 0.05020689, -1.50854897, -0.02487740, 1.68574724, 0.82497343]
    cal_20_mm = [-0.09984404, 0.14450927, 0.05021104, -1.48895958, -0.02498154, 1.66937602, 0.82479887]
    cal_22_mm = [-0.09978971, 0.14561315, 0.05021104, -1.48238086, -0.02502815, 1.66385170, 0.82474018]
    cal_24_mm = [-0.09973385, 0.14679426, 0.05021104, -1.47564609, -0.02507226, 1.65830631, 0.82469641]
    cal_30_mm = [-0.09939829, 0.15055267, 0.05021104, -1.45520355, -0.02522195, 1.64158327, 0.82452868]
    cal_40_mm = [-0.09887780, 0.15779365, 0.05021104, -1.41944408, -0.02554731, 1.61325706, 0.82427793]
    cal_50_mm = [-0.09827859, 0.16623325, 0.05020937, -1.38209379, -0.02595346, 1.58428060, 0.82405011]

  calibrated_0mm = cal_4_mm       # what joints for the floor, old=cal_2_mm
  calibrated_start = cal_14_mm    # what start position before grasping, can adjust, old=cal_12_mm

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

  if use_panda_threads:
    process = Process(target=franka_instance.move_joints, args=(target_state, speed_factor))
    process.start()
    process.join()
  else:
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

  if not depth_camera_connected and camera:
    rospy.logwarn("Depth camera is not connected, test aborted")
    return False
  elif not depth_camera_connected and not camera:
    rospy.logwarn("Depth camera set to FALSE, test proceeds")

  global testsaver, current_test_data, model, currently_testing, image_rate
  testsaver.enter_folder(request.name, forcecreate=True)
  current_test_data = GraspTestData() # wipe data clean
  current_test_data.start_test(request.name, model, get_depth_image, image_rate=image_rate)
  currently_testing = True

  # save the model in use if this file does not exist already
  savepath = testsaver.get_current_path()
  if not os.path.exists(savepath + "dqn_model" + testsaver.file_ext()):
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
      name = "luke-PC_13_15_A17"

    elif request.sensors == 1:
      group = "13-03-23"
      name = "luke-PC_17_23_A74"

    elif request.sensors == 2:
      group = "13-03-23"
      name = "luke-PC_17_23_A122"

    elif request.sensors == 3:
      group = "07-03-23"
      name = "luke-PC_13_37_A10"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 1:

    rospy.loginfo("Loading model for finger 1.0mm thick and 24mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      group = "31-03-23"
      name = "luke-PC_16_46_A90"

    elif request.sensors == 2:
      group = "27-03-23"
      name = "luke-PC_17_29_A142"

    elif request.sensors == 3:
      # group = "12-03-23"
      # name = "luke-PC_17:37_A220"
      # alternative training
      group = "06-04-23"
      name = "luke-PC_16_54_A217"

    else: rospy.logwarn(f"Sensors={request.sensors} not valid in load_baseline_model()")

  elif finger == 2:

    rospy.loginfo("Loading model for finger 1.0mm thick and 28mm width")

    if request.sensors == 0:
      # group = "16-01-23"
      # name = "luke-PC_14_21_A10"
      rospy.logwarn(f"Sensors={request.sensors} not yet added to baseline 3")

    elif request.sensors == 1:
      group = "31-03-23"
      name = "luke-PC_16_46_A104"

    elif request.sensors == 2:
      group = "27-03-23"
      name = "luke-PC_17_29_A170"
      
    elif request.sensors == 3:
      group = "10-03-23"
      name = "luke-PC_17_27_A239"

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
  if log_level > 0: rospy.loginfo(f"test heurisitc now set to TRUE")

  return []

def gentle_place(request=None):
  """
  Place an object gently below the gripper
  """

  # ensure there is no graspng going on
  cancel_grasping_callback()

  # what is the current gripper position
  [gx, gy, gz, h] = model.env.mj.get_state_metres(True) # realworld=True for real data

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

  if log_level > 0: rospy.loginfo("dqn node is publishing gripper demand for gentle place")
  demand_pub.publish(new_demand)

def move_panda_to(request=None):
  """
  Move the panda to a known joint position. These joint positions should
  be calibrated and tested by hand, do NOT run this function with untested
  joint positions
  """

  global move_panda
  if not move_panda:
    rospy.logwarn("move_panda_to() aborted as move_panda = False")
    return False

  if request.move_to_int == 0:
    # # regular calibrated spot
    # cal_0_mm = [-0.04460928, 0.38211982, -0.00579623, -1.20211819, -0.01439458, 1.61249259, 0.75540974]
    # cal_10_mm = [-0.04465586, 0.39663358, -0.00584523, -1.15490750, -0.01450791, 1.57477994, 0.75516820]
    # cal_12_mm = [-0.04481524, 0.40074865, -0.00589660, -1.14791929, -0.01460787, 1.56819993, 0.75505090]
    # cal_14_mm = [-0.04481524, 0.40295305, -0.00589637, -1.13912490, -0.01460929, 1.56178082, 0.75500080]
    # cal_20_mm = [-0.04469420, 0.41042913, -0.00585969, -1.11176795, -0.01456532, 1.54207928, 0.75484156]
    # cal_30_mm = [-0.04473575, 0.42504616, -0.00586498, -1.06289308, -0.01454851, 1.50777675, 0.75434994]
    # cal_40_mm = [-0.04478521, 0.44279476, -0.00585969, -1.00850484, -0.01452637, 1.47115869, 0.75380754]
    # cal_50_mm = [-0.04487317, 0.46457349, -0.00585969, -0.94686224, -0.01449903, 1.43148042, 0.75321650]

    # new calibrations 23/8/23 after robby, 4=initial touch, 0=v. firm, -2=fails
    cal_0_mm = [-0.10035520, 0.13415142, 0.05034569, -1.54992771, -0.02469004, 1.72361859, 0.82542563]
    cal_2_mm = [-0.10056089, 0.13577175, 0.05015793, -1.54504811, -0.02486288, 1.71835949, 0.82536323]
    cal_4_mm = [-0.10035362, 0.13717769, 0.05021119, -1.53977199, -0.02482598, 1.71273661, 0.82528626]
    cal_6_mm = [-0.10033584, 0.13799006, 0.05020785, -1.53366338, -0.02482757, 1.70740294, 0.82521534]
    cal_8_mm = [-0.10030479, 0.13879796, 0.05020785, -1.52739544, -0.02483351, 1.70205378, 0.82514384]
    cal_10_mm = [-0.10026742, 0.13964332, 0.05020785, -1.52111299, -0.02483833, 1.69663203, 0.82509060]
    cal_12_mm = [-0.10010887, 0.14053731, 0.05020618, -1.51491904, -0.02485414, 1.69118643, 0.82503248]
    cal_14_mm = [-0.10008372, 0.14145294, 0.05020689, -1.50854897, -0.02487740, 1.68574724, 0.82497343]
    cal_20_mm = [-0.09984404, 0.14450927, 0.05021104, -1.48895958, -0.02498154, 1.66937602, 0.82479887]
    cal_22_mm = [-0.09978971, 0.14561315, 0.05021104, -1.48238086, -0.02502815, 1.66385170, 0.82474018]
    cal_24_mm = [-0.09973385, 0.14679426, 0.05021104, -1.47564609, -0.02507226, 1.65830631, 0.82469641]
    cal_30_mm = [-0.09939829, 0.15055267, 0.05021104, -1.45520355, -0.02522195, 1.64158327, 0.82452868]
    cal_40_mm = [-0.09887780, 0.15779365, 0.05021104, -1.41944408, -0.02554731, 1.61325706, 0.82427793]
    cal_50_mm = [-0.09827859, 0.16623325, 0.05020937, -1.38209379, -0.02595346, 1.58428060, 0.82405011]

  elif request.move_to_int == 1:
    # top left
    cal_0_mm = [0.18512671, 0.37866549, 0.22363262, -1.21253592, -0.10893460, 1.60726003, 1.16956841]
    cal_10_mm = [0.18825695, 0.39035484, 0.22320580, -1.17424037, -0.11070185, 1.57652944, 1.16896099]
    cal_12_mm = [0.18910666, 0.39256941, 0.22320218, -1.16573255, -0.11115629, 1.57014674, 1.16883816]
    cal_14_mm = [0.18991396, 0.39483676, 0.22320218, -1.15689542, -0.11163938, 1.56374366, 1.16872168]
    cal_20_mm = [0.19228800, 0.40230689, 0.22320537, -1.13007192, -0.11322130, 1.54428125, 1.16835338]
    cal_30_mm = [0.19706933, 0.41643328, 0.22320842, -1.08209132, -0.11636666, 1.51035154, 1.16775696]
    cal_50_mm = [0.20903397, 0.45415981, 0.22322272, -0.96963235, -0.12516179, 1.43562458, 1.16632797]

  elif request.move_to_int == 2:
    # bottom left
    cal_0_mm = [0.23366993, -0.54271042, 0.37589574, -2.22000585, 0.17991284, 1.72729146, 1.34587936]
    cal_10_mm = [0.22811418, -0.54008458, 0.37610427, -2.19521192, 0.17898083, 1.70260116, 1.34520853]
    cal_12_mm = [0.22696416, -0.53974523, 0.37610188, -2.18974024, 0.17877689, 1.69760376, 1.34510382]
    cal_14_mm = [0.22584622, -0.53928485, 0.37610613, -2.18418006, 0.17853135, 1.69253790, 1.34499814]
    cal_20_mm = [0.22245999, -0.53770155, 0.37609984, -2.16731857, 0.17781672, 1.67756387, 1.34474422]
    cal_30_mm = [0.21712278, -0.53429485, 0.37609260, -2.13874013, 0.17637932, 1.65273764, 1.34451461]
    cal_50_mm = [0.20691718, -0.52499188, 0.37609244, -2.07961898, 0.17314562, 1.60336858, 1.34455698]

  elif request.move_to_int == 3:
    # bottom right
    cal_0_mm = [-0.00379845, -0.67043388, -0.54573473, -2.26318334, -0.32498345, 1.68728117, 0.48431597]
    cal_10_mm = [0.00731250, -0.66909944, -0.54599125, -2.23834650, -0.32438277, 1.66194544, 0.48656344]
    cal_12_mm = [0.00943737, -0.66909160, -0.54599602, -2.23279262, -0.32437264, 1.65686652, 0.48698529]
    cal_14_mm = [0.01170321, -0.66905026, -0.54599527, -2.22733633, -0.32434460, 1.65182822, 0.48742099]
    cal_20_mm = [0.01838100, -0.66819989, -0.54600097, -2.21056844, -0.32370427, 1.63669100, 0.48876885]
    cal_30_mm = [0.02929688, -0.66600868, -0.54592673, -2.18210780, -0.32251678, 1.61167038, 0.49064796]
    cal_50_mm = [0.05036132, -0.65874663, -0.54591255, -2.12315125, -0.31969950, 1.56254420, 0.49317398]

  elif request.move_to_int == 4:
    # top right
    cal_0_mm = [-0.05664975, 0.45664487, -0.51411564, -1.14189071, 0.22075273, 1.56880552, 0.34867809]
    cal_10_mm = [-0.06621081, 0.47048441, -0.51375980, -1.09982426, 0.22575926, 1.53636279, 0.34886103]
    cal_12_mm = [-0.06805916, 0.47280777, -0.51375451, -1.09054460, 0.22697677, 1.52964720, 0.34893845]
    cal_14_mm = [-0.07027538, 0.47558366, -0.51375982, -1.08090203, 0.22826196, 1.52281037, 0.34902136]
    cal_20_mm = [-0.07719636, 0.48447591, -0.51375645, -1.05102176, 0.23272260, 1.50179962, 0.34929511]
    cal_30_mm = [-0.09016613, 0.50185913, -0.51375853, -0.99682879, 0.24138361, 1.46464811, 0.34987587]
    cal_50_mm = [-0.12511534, 0.55004260, -0.51376929, -0.86303225, 0.26670520, 1.37924513, 0.35234891]

  else:
    rospy.logerr(f"move_panda_to() received unknown integer = {request.move_to_int}")
    return False
  
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

  if goto not in [0, 1, 2, 3, 4]:
    rospy.logerr(f"demo_movement() given invalid goto = {goto}")
    return False

  # go to the specified point
  move_msg.move_to_int = goto
  move_msg.height_mm = 50
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

    if grasp:
    
      # do a grasp
      execute_grasping_callback(reset=False)

      # wait for the grasp to execute or be stopped
      rospy.sleep(1)
      while continue_grasping:
        rospy.sleep(0.1)
        if not continue_demo or not continue_grasping:
          rospy.loginfo("Demo cancelled, demo_movement() is returning")
          return True
        elif rospy.is_shutdown(): return False

      # when grasp is done, lift to 50mm
      move_msg.move_to_int = goto
      move_msg.height_mm = 50
      moved = move_panda_to(move_msg)

      if not moved:
        rospy.logerr("Panda motion error in demo_movement(), aborting all")
        return False
      elif not continue_demo:
        rospy.loginfo("Demo cancelled, demo_movement() is returning")
        return True
      elif rospy.is_shutdown(): return False

  elif place:

    # lower before placing
    move_msg.move_to_int = goto
    move_msg.height_mm = 20
    moved = move_panda_to(move_msg)

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
    ["place", 0]
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

    if not success:
      rospy.logerr("Failed movement in loop_demo(), aborting all")
      return False
    
    demo_loop_int += 1

    if demo_loop_int >= len(loop_to_do):
      demo_loop_int = 0
    
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
  # global panda_reset_height_mm
  # panda_reset_height_mm = 0

  move_msg = PandaMoveToInt()

  if demo_panda_is_at in [0, 1, 2, 3, 4]:

    # lift the panda first
    move_msg.move_to_int = demo_panda_is_at
    move_msg.height_mm = 50
    moved = move_panda_to(move_msg)

    if not moved:
      rospy.logerr("Panda motion error in reset_demo(), aborting all")
      return False

  # now reset to the start position
  if reset_position not in [0, 1, 2, 3, 4]: reset_position = 0
  move_msg.move_to_int = reset_position
  move_msg.height_mm = 50
  moved = move_panda_to(move_msg)

  if not moved:
    rospy.logerr("Panda motion error in reset_demo(), aborting all")
    return False
  
  return True

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
  rospy.loginfo(f"  > move gripper is {move_gripper}")
  rospy.loginfo(f"  > move panda is {move_panda}")
  rospy.loginfo(f"  > using camera is {depth_camera_connected}")
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

  # publishers for displaying normalised nn input values
  norm_state_pub = rospy.Publisher(f"/{node_ns}/state", NormalisedState, queue_size=10)
  norm_sensor_pub = rospy.Publisher(f"/{node_ns}/sensor", NormalisedSensor, queue_size=10)
  debug_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)

  # user set - what do we load by default
  if use_devel:

    rospy.logwarn("Loading DEVEL policy")

    # load a devel training
    load = LoadModel()
    load.folderpath = "/home/luke/mujoco-devel/rl/models/dqn/"
    load.group_name = "24-07-23"
    load.run_name = "operator-PC_15:10_A2"
    load.run_id = None
    load_model(load)

    # now override action sizes: TESTING
    print(f"Action size gripper X: {model.env.mj.set.gripper_prismatic_X.value:.3f}")
    print(f"Action size gripper Y: {model.env.mj.set.gripper_revolute_Y.value:.3f}")
    print(f"Action size gripper Z: {model.env.mj.set.gripper_Z.value:.3f}")
    print(f"Action size base Z: {model.env.mj.set.base_Z.value:.3f}")
    model.env.mj.set.gripper_prismatic_X.value = 1.0e-3
    model.env.mj.set.gripper_revolute_Y.value = 0.01
    model.env.mj.set.gripper_Z.value = 2.0e-3
    model.env.mj.set.base_Z.value = 2.0e-3
  
  elif False:

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

  if log_level >= 3:
    # turn on all debug information
    model.env.log_level = 2
    model.env.mj.set.debug = True

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
  rospy.Service(f"/{node_ns}/place", Empty, gentle_place)
  rospy.Service(f"/{node_ns}/panda_move_to_int", PandaMoveToInt, move_panda_to)
  rospy.Service(f"/{node_ns}/start_demo", Empty, start_demo)
  rospy.Service(f"/{node_ns}/stop_demo", Empty, stop_demo)
  rospy.Service(f"/{node_ns}/reset_demo", Empty, reset_demo)
  rospy.Service(f"/{node_ns}/continue_demo", Demo, loop_demo)
  rospy.Service(f"/{node_ns}/set_reject_wrist_noise", SetBool, set_reject_wrist_noise_callback)

  try:
    while not rospy.is_shutdown(): rospy.spin() # and wait for service requests
  except Exception as e:
    end_test()
    rospy.logerror(f"dqn_node(...) failed, saved test data - exception is {e}")
  