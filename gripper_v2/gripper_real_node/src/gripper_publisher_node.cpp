#include "gripper_publisher.h"

int main(int argc, char **argv){
  
  ros::init(argc, argv, "gripper_publisher_node");
  ros::NodeHandle nh;

  // create an instance of the cw1 class
  GripperPublisher gripper(nh);

  // ros::AsyncSpinner spinner(4); // use 4 threads
  // spinner.start();
  // ros::waitForShutdown(); // this current thread now hangs

  // ros::MultiThreadedSpinner spinner(4); // use 4 threads
  // spinner.spin(); // this current thread now spings, so we have 5 threads total

  // loop at 100 Hz
  ros::Rate loop_rate(100);

  while (ros::ok()){
    ros::spinOnce(); // single-threaded
    loop_rate.sleep();
  }

  return 0;
}