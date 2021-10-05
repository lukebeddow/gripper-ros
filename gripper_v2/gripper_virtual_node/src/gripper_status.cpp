#include <gripper_virtual_node/gripper_status.h>

GripperVirtual::GripperVirtual() 
{
  /* Constructor */

  // subscriber to gripper demands
  demand_sub_ = nh_.subscribe("/gripper/demand", 10, &GripperVirtual::demand_callback, this);

  // subscribe to model joint states
  joints_sub_ = nh_.subscribe("/joint_states", 10, &GripperVirtual::joints_callback, this);

  // publisher for gripper status
  status_pub_ = nh_.advertise<gripper_virtual_node::gripper_msg>("/gripper/status", 10);

}

void GripperVirtual::set_y(gripper& state)
{
  /* This function converts the gazebo state of the gripper to a physical motor
  representation, ie from a prismatic+revolute to two prismatics */

  // distance between the two leadscrews
  static const float ls_distance = 35 * 0.001;

  // needs all in SI units! m and radians
  state.y = state.x + ls_distance * sin(state.th);
}

void GripperVirtual::demand_callback(const gripper_virtual_node::gripper_state msg) 
{
  /* Callback function for when gripper demand is updated */

  // save incoming message data
  gripper_demand_.x = msg.x;
  gripper_demand_.z = msg.z;
  gripper_demand_.th = msg.th;

  // calculate y leadscrew position
  set_y(gripper_demand_);

  ROS_ERROR("good job");
}

void GripperVirtual::joints_callback(const sensor_msgs::JointState msg)
{
  /* Callback function to save incoming joint state messages */

  // on the first loop, make a note of the key array indexes
  if (!updated_) {

    // find the size of the array since sizeof() gave only 24 bytes
    unsigned int loop_num = 0;
    for (double a : msg.position) {
      loop_num += 1;
    }

    // loop through every joint
    for (int i = 0; i < loop_num; i++) {
      if (msg.name[i] == "finger_1_prismatic_joint") {
        // save this index
        indexes_.push_back(i);
        break;
      }
    }
    for (int i = 0; i < loop_num; i++) {
      if (msg.name[i] == "finger_1_revolute_joint") {
        // save this index
        indexes_.push_back(i);
        break;
      }
    }
    for (int i = 0; i < loop_num; i++) {
      if (msg.name[i] == "palm_prismatic_joint") {
        // save this index
        indexes_.push_back(i);
        break;
      }
    }
    // only do this once
    updated_ = true;
  }

  // extract the joint positions from the incoming message
  for (int j = 0; j < indexes_.size(); j++) {
    if (j == 0) {
      gripper_state_.x = msg.position[indexes_[j]];
    }
    else if (j == 1) {
      gripper_state_.th = msg.position[indexes_[j]];
    }
    else if (j == 2) {
      gripper_state_.z = msg.position[indexes_[j]];
    }
  }

  // calculate the position of the y motor (front leadscrew)
  set_y(gripper_state_);
}

void GripperVirtual::publish()
{
  /* This function publishes a state message for the gripper */

  state_msg_.x = gripper_state_.x;
  state_msg_.y = gripper_state_.y;
  state_msg_.z = gripper_state_.z;
  state_msg_.th = gripper_state_.th;

  // check if the gripper has reached the demand
  static const float tol = 5e-4; // 0.5 mm

  if (abs(state_msg_.x - gripper_demand_.x) < tol and
      abs(state_msg_.y - gripper_demand_.y) < tol and
      abs(state_msg_.z - gripper_demand_.z) < tol) {
        
      state_msg_.is_target_reached = true;
  }
  else {
    state_msg_.is_target_reached = false;
  }

  status_pub_.publish(state_msg_);
}

void GripperVirtual::loop()
{
  /* This function loops to update the gripper status */

  ros::Rate r(50);

  while (ros::ok()) {
    // publish the gripper status
    if (updated_) {
      publish();
    }
    // check for callbacks
    ros::spinOnce();
    // loop at set rate
    r.sleep();
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "gripper_virtual_node");

  GripperVirtual vg;

  vg.loop();
}