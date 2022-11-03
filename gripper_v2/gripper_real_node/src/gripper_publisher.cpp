#include "gripper_publisher.h"

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
  gauge1 = msg.gauge1;
  gauge2 = msg.gauge2;
  gauge3 = msg.gauge3;
  gauge4 = msg.gauge4;

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

  return success;
}

gripper_msgs::GripperState Gripper_ROS::to_state_msg()
{
  /* return a state message for this gripper */

  gripper_msgs::GripperState state_msg;

  // fill with data
  state_msg.units = "m_rad";
  state_msg.pose.x = get_x_m();
  state_msg.pose.y = get_y_m();
  state_msg.pose.z = get_z_m();
  state_msg.angle = th;

  state_msg.step.x = step.x;
  state_msg.step.y = step.y;
  state_msg.step.z = step.z;

  state_msg.speed.x = speed.x;
  state_msg.speed.y = speed.y;
  state_msg.speed.z = speed.z;

  state_msg.sensor.gauge1 = gauge1;
  state_msg.sensor.gauge2 = gauge2;
  state_msg.sensor.gauge3 = gauge3;
  state_msg.sensor.gauge4 = gauge4;

  state_msg.ftdata = last_ft_data;

  // ROS_INFO_STREAM("last ft data z is " << state_msg.ftdata.force.z);

  state_msg.is_target_reached = is_target_reached;
  state_msg.is_power_saving = is_power_saving;
  state_msg.is_stopped = is_stopped;

  state_msg.timestamp = ros::Time::now().toSec();

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
  force_torque_sub_ = nh_.subscribe("gripper/real/ftsensor", 10,
    &GripperPublisher::ftsensor_callback, this);

  // set up publishers
  sensor_pub_ = nh_.advertise<gripper_msgs::SensorState>("gripper/real/sensors", 10);
  state_pub_ = nh_.advertise<gripper_msgs::GripperState>("gripper/real/state", 10);
  input_pub_ = nh_.advertise<gripper_msgs::GripperInput>("gripper/real/input", 10);
  demand_pub_ = nh_.advertise<gripper_msgs::GripperDemand>("gripper/demand", 10);

  // setup service
  dqn_srv_ = nh_.serviceClient<gripper_msgs::ControlRequest>("/gripper/control/dqn");
}

void GripperPublisher::publish_demand(std::vector<float> target_state)
{
  /* publish a demand */

  if (target_state.size() == 0) {
    return;
  }

  gripper_msgs::GripperDemand demand;

  demand.state.pose.x = target_state[0];
  demand.state.pose.y = target_state[1];
  demand.state.pose.z = target_state[2];

  // demand_pub_.publish(demand); // this is very slow!
  demand_callback(demand);

  ROS_INFO_STREAM("Demand published by gripper_publisher of (x, y, z) mm = (" << target_state[0] * 1e3
    << ", " << target_state[1] * 1e3 << ", " << target_state[2] * 1e3 << ")");
}

void GripperPublisher::demand_callback(const gripper_msgs::GripperDemand& msg)
{
  /* get a user specified demand, check it, and publish it */

  ROS_INFO("Received a demand in gripper_publisher");

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
  sensor_pub_.publish(new_state.sensor);

  // TESTING: not using service method
  return;

  // if the target has not been reached, do nothing
  if (new_state.is_target_reached == false) return;

  // otherwise, request a new control signal
  gripper_msgs::ControlRequest srv_msg;

  srv_msg.request.gripper_state.push_back(new_state.pose.x);
  srv_msg.request.gripper_state.push_back(new_state.pose.y);
  srv_msg.request.gripper_state.push_back(new_state.pose.z);

  srv_msg.request.gauge1.push_back(last_state.sensor.gauge1);
  srv_msg.request.gauge1.push_back(new_state.sensor.gauge1 - last_state.sensor.gauge1);
  srv_msg.request.gauge1.push_back(new_state.sensor.gauge1);

  srv_msg.request.gauge2.push_back(last_state.sensor.gauge2);
  srv_msg.request.gauge2.push_back(new_state.sensor.gauge2 - last_state.sensor.gauge2);
  srv_msg.request.gauge2.push_back(new_state.sensor.gauge2);

  srv_msg.request.gauge3.push_back(last_state.sensor.gauge3);
  srv_msg.request.gauge3.push_back(new_state.sensor.gauge3 - last_state.sensor.gauge3);
  srv_msg.request.gauge3.push_back(new_state.sensor.gauge3);

  bool success = dqn_srv_.call(srv_msg);

  if (not success) return;

  ROS_INFO("Control request finished");

  last_state = new_state;

  publish_demand(srv_msg.response.target_state);
}

void GripperPublisher::ftsensor_callback(const geometry_msgs::Wrench& msg)
{
  /* receive data from force-torque sensor */

  // copy data across
  gripper_.last_ft_data.force.x = msg.force.x;
  gripper_.last_ft_data.force.y = msg.force.y;
  gripper_.last_ft_data.force.z = msg.force.z;
  gripper_.last_ft_data.torque.x = msg.torque.x;
  gripper_.last_ft_data.torque.y = msg.torque.y;
  gripper_.last_ft_data.torque.z = msg.torque.z; 
}
