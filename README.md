# luke_gripper

This repository contains ros and gazebo integration for my gripper design.

The gazebo_models folder contains models which should be added to .gazebo (these may be out of date).

The remaining folders are ros packages which can be built with: $ catkin build

gripper_v2 contains the robot model, both the Franka Emika Panda arm and my gripper, as well as packages to configure it for use with moveit and gazebo.

grasp_test is a package to communicate with moveit and pass instructions to the robot.

gz_link is a package to communicate between gazebo and ros.
