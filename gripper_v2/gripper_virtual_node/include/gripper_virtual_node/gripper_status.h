#include <ros/ros.h>
#include <gripper_virtual_node/gripper_state.h>
#include <gripper_virtual_node/gripper_msg.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>

class GripperVirtual
{
public:

  struct gripper {
    double x = -1;
    double y = -1;
    double z = -1;
    double th = 0;
  };

  /* member functions */
  GripperVirtual();
  void set_y(gripper& state);
  void demand_callback(const gripper_virtual_node::gripper_state msg);
  void joints_callback(const sensor_msgs::JointState msg);
  void publish();
  void loop();

  /* member variables */
  ros::NodeHandle nh_;
  ros::Subscriber demand_sub_;
  ros::Subscriber joints_sub_;
  ros::Publisher status_pub_;

  gripper_virtual_node::gripper_msg state_msg_;

  gripper gripper_state_;
  gripper gripper_demand_;

  std::vector<int> indexes_;
  std::vector<std_msgs::String> names_;

  bool updated_ = false;
};