// include guards, prevent .h file being defined multiple times (linker error)
#ifndef GRIPPER_PUBLISHER_H_
#define GRIPPER_PUBLISHER_H_

// ros includes
#include <ros/ros.h>
#include <std_msgs/Float64.h>

// gripper messages
#include "gripper_msgs/pose.h"
#include "gripper_msgs/status.h"
#include "gripper_msgs/GripperDemand.h"
#include "gripper_msgs/GripperState.h"
#include "gripper_msgs/GripperOutput.h"
#include "gripper_msgs/GripperInput.h"
#include "gripper_msgs/ControlRequest.h"
#include "gripper_msgs/SensorRaw.h"

// this package includes
#include "gripper.h"

using luke::Gripper;

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

  constexpr static int buffer_size = 50;

  constexpr static float g1_offset = 0.70e6;
  constexpr static float g2_offset = -0.60e6;
  constexpr static float g3_offset = 0.56e6;

  constexpr static float g1_scale = 1.258e-6;
  constexpr static float g2_scale = 1.258e-6;
  constexpr static float g3_scale = 1.258e-6;

  constexpr static float normalisation_value = 2;


  // // what we already have from gripper.h
  // double x, y, z;
  // double th;
  // struct { int x, y, z; } step;

  // uncalibrated sensor data
  long gauge1 = 0;
  long gauge2 = 0;
  long gauge3 = 0;

  // is the gripper at the target
  bool is_target_reached = false;

  // speed settings of the motors
  struct { 
    float x = 150;
    float y = 150;
    float z = 150;
  } speed;

  // optional settings default start state
  bool is_stopped = false;
  bool is_power_saving = true;
  bool debug = false;

};

class GripperPublisher
{

public:

  /* ----- class member functions ----- */

  GripperPublisher(ros::NodeHandle nh);
  void save_gauge_data(double g1, double g2, double g3);
  void demand_callback(const gripper_msgs::GripperDemand& msg);
  void state_callback(const gripper_msgs::GripperOutput& msg);
  void publish_demand(std::vector<float> target_state);

  /* ----- class member variables ----- */

  // gripper state
  Gripper_ROS gripper_;

  // state at last control interval
  gripper_msgs::GripperState last_state;

  // ros functionality
  ros::NodeHandle nh_;
  ros::Publisher sensor_pub_;
  ros::Publisher input_pub_;
  ros::Publisher state_pub_;
  ros::Publisher demand_pub_;
  ros::Subscriber demand_sub_;
  ros::Subscriber output_sub_;
  ros::ServiceClient dqn_srv_;

};

#endif // GRIPPER_PUBLISHER_H_