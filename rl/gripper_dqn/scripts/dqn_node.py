#!/usr/bin/env python3

import rospy
import sys
from gripper_msgs.srv import ControlRequest, ControlRequestResponse
import networks

from time import sleep

# insert the mymujoco path
sys.path.insert(0, "/home/luke/mymujoco/rl")

# create model instance
from TrainDQN import TrainDQN
model = TrainDQN(use_wandb=False, no_plot=True, log_level=1)

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
  sleep(3)
  rospy.loginfo("Finished sleeping")

  return res

  # srv.Response.target_state = srv.Request.gripper_state[:]
  # motor_nudge = 100e-3
  # base_nudge = 2e-3

  # # apply the action
  # if action.item == 0:
  #   srv.Response.target_state[0] += motor_nudge
  # elif action.item == 1:
  #   srv.Response.target_state[0] -= motor_nudge
    

if __name__ == "__main__":

  # now initilise ros
  rospy.init_node("dqn_node")

  # load the file that is local
  net = networks.DQN_2L60
  model.init(net)
  model.load(id=None, foldername="/home/luke/gripper_repo_ws/src/rl/gripper_dqn/scripts/models", 
             folderpath="")

  # create service responder
  rospy.Service("/gripper/control/dqn", ControlRequest, srv_callback)
  

  # # create raw data publishers
  # connected_pub = rospy.Publisher("/gripper/real/connected", Bool, queue_size=10)
  # gauge1_pub = rospy.Publisher("/gripper/real/gauge1", Float64, queue_size=10)
  # gauge2_pub = rospy.Publisher("/gripper/real/gauge2", Float64, queue_size=10)
  # gauge3_pub = rospy.Publisher("/gripper/real/gauge3", Float64, queue_size=10)

  # # create data transfer input/output
  # rospy.Subscriber("/gripper/real/input", GripperInput, demand_callback)
  # state_pub = rospy.Publisher("/gripper/real/output", GripperOutput, queue_size=10)

  rate = rospy.Rate(10)

  while not rospy.is_shutdown():

    # # get the most recent state of the gripper
    # state = mygripper.update_state()

    # # check if the connection is live yet
    # connected_pub.publish(mygripper.connected)

    # if mygripper.connected:

    #   # fill in the gripper output message
    #   output_msg = state_to_msg(state)

    #   # publish data
    #   state_pub.publish(output_msg)
    #   gauge1_pub.publish(state.gauge1_data)
    #   gauge2_pub.publish(state.gauge2_data)
    #   gauge3_pub.publish(state.gauge3_data)

    rate.sleep()