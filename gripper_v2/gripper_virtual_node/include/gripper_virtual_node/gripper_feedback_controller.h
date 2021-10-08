#pragma once

#include <ros/ros.h>

#include <gripper_virtual_node/gripper_state.h>
#include <gripper_virtual_node/gripper_msg.h>

class GraspController 
{
  /* This class is a controller for the virtual gripper in Gazebo, which aims
  to adjust the gripper fingers and maybe nudge the arm position to achieve
  grasping autonomously */

public:
  /* member functions */
  GraspController();

  /* member variables */
  ros::NodeHandle nh;

}