# luke-gripper-ros

This repository contains Robot Operating System (ROS) integration for my gripper design.
* rl/ contains reinforcement learning packages, with the main implementation in ```gripper_dqn```, which can implement any RL algorithm depending on the model, not just DQN.
* gripper_v2 contains the gripper interface packages (```gripper_real_node/gripper_msgs```), robot models (```gripper_description```), packages to configure it for use with moveit and gazebo, and more.
* grasp_test is depreciated, and contains old development scripts.
* gz_link is depreciated, and contains old gazebo simulator plugins.

Launch the main reinforcement learning grasping pipeline with ```roslaunch gripper_dqn rl.launch```. However, this has the following dependencies:
* Loading and evaluating models requires the codebase https://github.com/RPL-CS-UCL/luke-gripper-mujoco to be accessible in Python.
* A saved model and compatible compilation of the above codebase must also be accessible.
* The hardware must be configured, for best results connect the gripper to power first, then plug the USB-C cable into the computer.
* A compiled version of https://github.com/RPL-CS-UCL/franka_interface, with libranka.so exposed, must be accessible in Python.

Deploying a trained model for grasping will require examination of ```rl/gripper_dqn/scripts/rl_grasping_node.py```, which contains important paths to all of the above items.

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
