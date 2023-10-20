# luke-gripper-ros

This repository contains ros and gazebo integration for my gripper design.

The gazebo_models folder contains models which should be added to .gazebo (these may be out of date).

The remaining folders are ros packages which can be built with: $ catkin build

gripper_v2 contains the robot model, both the Franka Emika Panda arm and my gripper, as well as packages to configure it for use with moveit and gazebo.

grasp_test is a package to communicate with moveit and pass instructions to the robot.

gz_link is a package to communicate between gazebo and ros.

## no catkin build

If catkin_make works but there is no catkin build:

```
sudo apt install python3-catkin-tools python3-osrf-pycommon
```

## error with ROS packages eg 'joint_trajectory_controller'

Ensure the ROS dependencies are updated

```
rosdep update
rosdep install --ignore-src --from-paths src -y -r
```

## libfranka.so not found

Make sure the libfranka.so file is exposed, you can export it or add this line to .bashrc:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/path/to/franka_interface/third_party/libfranka/lib
```
