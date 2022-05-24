#!/bin/bash

# a script to generate mujoco source files

# set variables (note: no spaces allowed around = sign)
WS_DIR=/home/luke/gripper_repo_ws
MUJOCO_DIR=src/gripper_v2/gripper_description/urdf/mujoco

# if no arguments supplied
if [ $# -eq 0 ]
  then
    # run both subscripts
    echo "No arguments supplied, run everything"
    source $WS_DIR/$MUJOCO_DIR/run_xacro.sh
    source $WS_DIR/$MUJOCO_DIR/run_mjcf.sh
fi

# if given argument first argument == 1
if [ "$1" -eq 1 ]
  then
    # only run the second subscript
    echo "Argument 1 supplied, only recompile mjcf"
    source $WS_DIR/$MUJOCO_DIR/run_mjcf.sh
fi



# ----- old code below, now split into the above two scripts ----- #

# # set variables (note: no spaces allowed around = sign)
# WS_DIR=/home/luke/gripper_repo_ws
# MUJOCO_DIR=src/gripper_v2/gripper_description/urdf/mujoco/

# # make sure we source ros
# cd $WS_DIR
# source devel/setup.bash

# # move to the mujoco directory
# cd $WS_DIR/$MUJOCO_DIR

# # generate the urdf models
# roslaunch gripper_scripting mujoco.launch

# # remove the old mujoco xml files
# rm -f gripper_mujoco.xml panda_mujoco.xml panda_and_gripper_mujoco.xml gripper_task.xml

# # compile new mujoco xml files
# ./compile gripper_mujoco.urdf gripper_mujoco.xml
# ./compile panda_mujoco.urdf panda_mujoco.xml
# ./compile panda_and_gripper_mujoco.urdf panda_and_gripper_mujoco.xml
# ./compile gripper_task.urdf gripper_task.xml

# # add extra features to the xml files
# ./xml_script.py