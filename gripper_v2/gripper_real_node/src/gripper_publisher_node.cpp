#include "gripper_real_node/gripper_publisher.h"

int main(int argc, char **argv){
  
  ros::init(argc, argv, "gripper_publisher_node");
  ros::NodeHandle nh;

  // create an instance of the cw1 class
  GripperPublisher gripper(nh);

  // loop at 100 Hz
  ros::Rate loop_rate(100);

  while (ros::ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}