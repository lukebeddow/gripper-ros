#include <grasp_test.h>

int main(int argc, char** argv)
{
  // initialise ros and the node
  ros::init(argc, argv, "run_grasp_test");
  ros::NodeHandle nh("~");
  
  // MoveIt! requirement for non-blocking group.move()
  ros::AsyncSpinner spinner(1);
  spinner.start();

  // create grasp_test object
  GraspTest g_test(nh);

  // loop rate in Hz
  ros::Rate rate(2);

  while (ros::ok ()) {

    ros::spinOnce();
    rate.sleep();
  }
}

