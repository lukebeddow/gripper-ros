#include <gz_link.h>

namespace gazebo {
  // namespace PluginTest {

    PluginTest::PluginTest() : WorldPlugin()
    {
      /* constructor */
    }

    void PluginTest::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) {
    
      // Make sure the ROS node for Gazebo has already been initialized
      if (!ros::isInitialized())
      {
        ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin. "
          << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in the gazebo_ros package)");
        return;
      }

      nh = ros::NodeHandle("PluginTest");
      // pub = nh.advertise<geometry_msgs::PoseStamped>("/PluginTest/pose", 10);
      pub = nh.advertise<std_msgs::String>("/PluginTest/string", 10);

      updateConnection = event::Events::ConnectWorldUpdateBegin(
            boost::bind(&PluginTest::getInput, this));
    }

    bool PluginTest::kbhit()
    {
      /* function to get a keypress without blocking */
      struct termios oldt, newt;
      int ch;
      int oldf;
      
      tcgetattr(STDIN_FILENO, &oldt);
      newt = oldt;
      newt.c_lflag &= ~(ICANON | ECHO);
      tcsetattr(STDIN_FILENO, TCSANOW, &newt);
      oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
      fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
      
      ch = getchar();
      
      tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
      fcntl(STDIN_FILENO, F_SETFL, oldf);
      
      if(ch != EOF)
      {
        ungetc(ch, stdin);
        return true;
      }
      
      return false;
    }

    void PluginTest::getInput()
    {
  
      if (kbhit()) {

        int ch = getchar();
        pubString.data.clear();
        pubString.data.push_back(ch);

        // publish this character
        pub.publish(pubString);

        if (ch == 't') {
          // ROS_INFO("Testing worked!");
        }

        if (ch == 'p') {

          // // publish the target pose
          // targetPose.header.frame_id = "base";
          // targetPose.header.stamp = ros::Time::now();
          // pub.publish(targetPose);
          // ROS_INFO("targetPose has been published to ROS");
        }
      }
      
    }

  // }

  // register plugin with gazebo
  GZ_REGISTER_WORLD_PLUGIN(PluginTest);
}

