#!/bin/bash

# move to the mujoco directory
cd $WS_DIR/$MUJOCO_DIR

# # run xacro to create the mjcf includes
# cd mjcf_include
# ./run_xacro.sh
# cd ..

# remove the old mujoco xml files
rm -f gripper_mujoco.xml panda_mujoco.xml panda_and_gripper_mujoco.xml gripper_task.xml

# compile new mujoco xml files
./compile gripper_mujoco.urdf gripper_mujoco.xml
./compile panda_mujoco.urdf panda_mujoco.xml
./compile panda_and_gripper_mujoco.urdf panda_and_gripper_mujoco.xml
./compile gripper_task.urdf gripper_task.xml

# add extra features to the xml files
./xml_script.py