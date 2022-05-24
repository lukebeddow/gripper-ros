#!/bin/bash

# generate xacro files using ros

# # set variables (note: no spaces allowed around = sign)
# WS_DIR=/home/luke/gripper_repo_ws
# MUJOCO_DIR=src/gripper_v2/gripper_description/urdf/mujoco/

# make sure we source ros
cd $WS_DIR
source devel/setup.bash

# move to the mujoco directory
cd $WS_DIR/$MUJOCO_DIR

# generate the urdf models
roslaunch gripper_scripting mujoco.launch