// include guards, prevent .h file being defined multiple times (linker error)
#ifndef GRIPPER_PUBLISHER_H_
#define GRIPPER_PUBLISHER_H_

// system includes
#include <ros/ros.h>

// gripper messages
#include "gripper_msgs/pose.h"
#include "gripper_msgs/status.h"
#include "gripper_msgs/GripperDemand.h"
#include "gripper_msgs/GripperState.h"
#include "gripper_msgs/GripperOutput.h"
#include "gripper_msgs/GripperInput.h"

// this package includes
#include "gripper_real_node/gripper.h"

class Gripper_ROS: public Gripper
{
  /* gripper.h class extension to add ROS functionality and to interface better
  with the real gripper */

public:

  /* ----- class member functions ----- */

  Gripper_ROS();
  bool from_output_msg(gripper_msgs::GripperOutput msg);
  gripper_msgs::GripperState to_state_msg();

  /* ----- class member variables ------ */

  // // what we already have from gripper.h
  // double x, y, z;
  // double th;
  // struct { int x, y, z; } step;

  // sensor data
  long gauge1 {};
  long gauge2 {};
  long gauge3 {};

  // extra variables reflecting settings on the real gripper
  bool is_target_reached = false;
  bool is_stopped = false;
  bool is_power_saving = true;  // default
};

class GripperPublisher
{

public:

  /* ----- class member functions ----- */

  GripperPublisher(ros::NodeHandle nh);
  void demand_callback(const gripper_msgs::GripperDemand& msg);
  void state_callback(const gripper_msgs::GripperOutput& msg);

  /* ----- class member variables ----- */

  // gripper state
  Gripper_ROS gripper_;

  // ros functionality
  ros::NodeHandle nh_;
  ros::Publisher input_pub_;
  ros::Publisher state_pub_;
  ros::Subscriber demand_sub_;
  ros::Subscriber output_sub_;

};

#endif // GRIPPER_PUBLISHER_H_