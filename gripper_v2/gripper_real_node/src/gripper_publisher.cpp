#include "gripper_real_node/gripper_publisher.h"

Gripper_ROS::Gripper_ROS()
{
  /* constructor */
}

bool Gripper_ROS::from_output_msg(gripper_msgs::GripperOutput msg)
{
  /* update the gripper state based on an output message */

  // extract data
  is_target_reached = msg.is_target_reached;
  float x = msg.motor_x_m;
  float y = msg.motor_y_m;
  float z = msg.motor_z_m;
  long g1 = msg.gauge1;
  long g2 = msg.gauge2;
  long g3 = msg.gauge3;

  // set the new state
  bool success = set_xyz_m(x, y, z);

  // if the state is outside limits
  if (not success) {
    ROS_WARN_STREAM("Reported real world gripper state is (x, y, z) in mm: "
      << x * 1e3 << ", " << y * 1e3 << ", " << z * 1e3);
    ROS_WARN_STREAM("Internal computer gripper state is capped at (x, y, z) in mm: "
      << get_x_mm() << ", " << get_y_mm() << ", " << get_z_mm());
    ROS_ERROR("Real gripper state received is outside safe limits in gripper.h");
  }

  // do we scale the raw gauge data here?
  gauge1 = g1;
  gauge2 = g2;
  gauge3 = g3;

  return success;
}

gripper_msgs::GripperState Gripper_ROS::to_state_msg()
{
  /* return a state message for this gripper */

  gripper_msgs::GripperState state_msg;

  // fill with data
  state_msg.pose.x = get_x_m();
  state_msg.pose.y = get_y_m();
  state_msg.pose.z = get_z_m();

  state_msg.step.x = step.x;
  state_msg.step.y = step.y;
  state_msg.step.z = step.z;

  state_msg.sensor.gauge1 = gauge1;
  state_msg.sensor.gauge2 = gauge2;
  state_msg.sensor.gauge3 = gauge3;

  state_msg.angle = th;
  state_msg.is_target_reached = is_target_reached;
  state_msg.is_power_saving = is_power_saving;
  state_msg.is_stopped = is_stopped;

  return state_msg;
}

GripperPublisher::GripperPublisher(ros::NodeHandle nh)
{
  /* constructor */

  nh_ = nh;

  // set up subscribers
  demand_sub_ = nh_.subscribe("gripper/demand", 10, 
    &GripperPublisher::demand_callback, this);
  output_sub_ = nh_.subscribe("gripper/real/output", 10,
    &GripperPublisher::state_callback, this);

  // set up publishers
  state_pub_ = nh_.advertise<gripper_msgs::GripperState>("gripper/real/state", 10);
  input_pub_ = nh_.advertise<gripper_msgs::GripperInput>("gripper/real/input", 10);

}

void GripperPublisher::demand_callback(const gripper_msgs::GripperDemand& msg)
{
  /* get a user specified demand, check it, and publish it */

  bool success = false;
  float x = msg.state.pose.x;
  float y = msg.state.pose.y;
  float z = msg.state.pose.z;
  float th = msg.state.angle;

  // if this is a state command
  if (not msg.ignore_state) {

    // set the demanded gripper state based on the user selected units
    if (msg.angle_demand) {
      if (msg.use_mm and msg.use_deg) {
        success = gripper_.set_xyz_mm_deg(x, th, z);
      }
      else if (msg.use_mm and not msg.use_deg) {
        success = gripper_.set_xyz_mm_rad(x, th, z);
      }
      else if (not msg.use_mm and msg.use_deg) {
        success = gripper_.set_xyz_m_deg(x, th, z);
      }
      else {
        success = gripper_.set_xyz_m_rad(x, th, z);
      }
    }
    else {
      if (msg.use_mm) {
        success = gripper_.set_xyz_mm(x, y, z);
      }
      else {
        success = gripper_.set_xyz_m(x, y, z);
      }
    }
  }
  
  // if the demand falls outside the safe limits
  if (not success) {
    ROS_WARN("Gripper demand outside safe limits, has been clipped\n");
  }

  // prepare publish the demand message
  gripper_msgs::GripperInput new_input;

  // copy over demand information
  new_input.x_m = gripper_.get_x_m();
  new_input.y_m = gripper_.get_y_m();
  new_input.z_m = gripper_.get_z_m();
  new_input.home = msg.home;
  new_input.ignore_xyz_command = msg.ignore_state;

  // work out if we need to make adjustments to settings
  if (gripper_.is_power_saving != msg.state.is_power_saving) {
    if (msg.state.is_power_saving) {
      new_input.power_saving_on = true;
    }
    else {
      new_input.power_saving_off = true;
    }
  }
  if (gripper_.is_stopped != msg.state.is_stopped) {
    if (msg.state.is_stopped) {
      new_input.stop = true;
    }
    else {
      new_input.resume = true;
    }
  }

  // save that we have adjusted the settings
  gripper_.is_power_saving = msg.state.is_power_saving;
  gripper_.is_stopped = msg.state.is_stopped;

  // publish
  input_pub_.publish(new_input);
}

void GripperPublisher::state_callback(const gripper_msgs::GripperOutput& msg)
{
  /* get a gripper reported state, check it, and publish */

  // we have a new state, update data
  gripper_.from_output_msg(msg);

  // publish
  gripper_msgs::GripperState new_state = gripper_.to_state_msg();
  state_pub_.publish(new_state);

}