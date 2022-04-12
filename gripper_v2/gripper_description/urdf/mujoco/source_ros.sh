#!/bin/bash

WS_DIR=/home/luke/gripper_repo_ws
MUJOCO_DIR=src/gripper_v2/gripper_description/urdf/mujoco/

# source ros
cd $WS_DIR
source devel/setup.bash
cd $WS_DIR/$MUJOCO_DIR

# roscore in the background
roscore &

# set the flag to know ros is sourced
export LUKE_FLAG=1

# if first arg is set, recall make
if [ -n "$1" ]; then
  $1 $2
fi