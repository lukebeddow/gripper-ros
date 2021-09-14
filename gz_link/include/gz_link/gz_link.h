#ifndef GZ_LINK_H_
#define GZ_LINK_H_

#include <gazebo/gazebo.hh>
#include <ros/ros.h>

#include <boost/bind.hpp>

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>

namespace gazebo
{
  // custom class inherits from predfined gazebo template
  class PluginTest : public WorldPlugin
  {
  public: 
    PluginTest();
    void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);
    bool kbhit();
    void getInput();
  };

  /* Member variables */
  event::ConnectionPtr updateConnection;
  ros::NodeHandle nh;
  ros::Publisher pub;
  geometry_msgs::PoseStamped targetPose;
  std_msgs::String pubString;
}

#endif