#!/usr/bin/env python3

import rospy
import sys
from gripper_msgs.srv import ControlRequest, ControlRequestResponse
import networks

from gripper_msgs.msg import GripperInput, GripperOutput, GripperState, GripperDemand, MotorState

# insert the mymujoco path
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
model = TrainDQN(use_wandb=False, no_plot=True, log_level=1, device="cpu")

# globals
gauge_actual_pub = None
prev_g1 = 0
prev_g2 = 0
prev_g3 = 0

new_demand = False
demand = GripperInput()

ready_for_new_action = False

def state_callback(data):
  """
  Checks whether the target has been reached, and if so triggers new demand
  """

  if data.is_target_reached:

    state_vec = [
      data.motor_x_m,
      data.motor_y_m,
      data.motor_z_m,
      data.gauge1,
      data.gauge2,
      data.gauge3
    ]

    state_vec = model.to_torch(state_vec)

    # use the state to predict the best action (test=True means pick best possible)
    action = model.select_action(state_vec, decay_num=1, test=True)

    # set the service response with the new position this action results in
    new_target_state = model.env.mj.set_action(action.item())

    # create a demand for this new target state
    demand.x = new_target_state[0]
    demand.y = new_target_state[1]
    demand.z = new_target_state[2]
    demand.command_type = "command"

    new_demand = True

def srv_callback(srv):
  """
  Receives a gripper control request and replies with a new state
  """

  # extract current state and convert to pytorch vector
  state = model.to_torch(
    list(srv.gripper_state) 
    + list(srv.gauge1) 
    + list(srv.gauge2)
    + list(srv.gauge3)
  )

  # use the state to predict the best action (test=True means pick best possible)
  action = model.select_action(state, decay_num=1, test=True)

  # set the service response with the new position this action results in
  new_target_state = model.env.mj.set_action(action.item())

  res = ControlRequestResponse()

  res.target_state = new_target_state

  rospy.loginfo("Sleeping now")
  rospy.sleep(0.3) # Sleeps for 1 sec
  rospy.loginfo("Finished sleeping")

  return res

def data_callback(state):
  """
  Receives new state data for the gripper
  """
  
  state_vec = [
    state.pose.x, state.pose.y, state.pose.z
  ]

  sensor_vec = [
    state.sensor.gauge1, state.sensor.gauge2, state.sensor.gauge3
  ]
  
  timestamp = 0 # not using this for now

  model.env.mj.input_real_data(state_vec, sensor_vec, timestamp)

  if state.is_target_reached:

    global ready_for_new_action
    ready_for_new_action = True

    # for testing - visualise the actual gauge network inputs
    vec = model.env.mj.get_finger_gauge_data()
    global gauge_actual_pub
    mymsg = MotorState()
    mymsg.x = vec[0]
    mymsg.y = vec[1]
    mymsg.z = vec[2]
    gauge_actual_pub.publish(mymsg)

def generate_action():
  """
  Get a new gripper state
  """

  rospy.loginfo("generating a new action")

  obs = model.env.mj.get_real_observation()
  obs = model.to_torch(obs)
  
  # use the state to predict the best action (test=True means pick best possible)
  action = model.select_action(obs, decay_num=1, test=True)

  # set the service response with the new position this action results in
  new_target_state = model.env.mj.set_action(action.item())

  rospy.loginfo("Sleeping now")
  rospy.sleep(0.5) # Sleeps for 1 sec
  rospy.loginfo("Finished sleeping")

  return new_target_state

if __name__ == "__main__":

  # now initilise ros
  rospy.init_node("dqn_node")

  # load the file that is local
  net = networks.DQN_2L60
  model.init(net)
  folderpath = "/home/luke/mymujoco/rl/models/dqn/09-06-22/"
  foldername = "luke-PC_16_01_A2"
  model.load(id=None, folderpath=folderpath, foldername=foldername)
  # model.load(id=None, foldername="/home/luke/gripper_repo_ws/src/rl/gripper_dqn/scripts/models", 
  #            folderpath="")

  # create service responder
  # rospy.Service("/gripper/control/dqn", ControlRequest, srv_callback)
  
  rospy.Subscriber("/gripper/real/state", GripperState, data_callback)
  demand_pub = rospy.Publisher("/gripper/demand", GripperDemand, queue_size=10)

  gauge_actual_pub = rospy.Publisher("/gripper/dqn/gauges", MotorState, queue_size=10)
  # rospy.Subscriber("/gripper/real/output", GripperOutput, state_callback)
  # demand_pub = rospy.Publisher("/gripper/real/input", GripperInput, queue_size=10)

  rate = rospy.Rate(20)

  while not rospy.is_shutdown():

    if ready_for_new_action:

      new_target_state = generate_action()

      new_demand = GripperDemand()

      new_demand.state.pose.x = new_target_state[0]
      new_demand.state.pose.y = new_target_state[1]
      new_demand.state.pose.z = new_target_state[2]

      rospy.loginfo("dqn node is publishing a new demand")
      demand_pub.publish(new_demand)

    rate.sleep()