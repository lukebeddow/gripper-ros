#pragma once

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>

#include <string>
#include <vector>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// #include <moveit_msgs/GetPlanningScene.h>
// #include <moveit/robot_model_loader/robot_model_loader.h>
// #include <moveit/kinematic_constraints/utils.h>
// #include <moveit/planning_scene/planning_scene.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

class GraspTest
{
public:

  // FOR TESTING
  // box structure from which to make collision objects
    struct Box {

      double x, y, z;
      double roll, pitch, yaw;
      double length;
      double width;
      double height;

      // constructor
      Box(double x, double y, double z, double roll, double pitch, double yaw,
          double length, double width, double height)
          : x(x), y(y), z(z), roll(roll), pitch(pitch), yaw(yaw),
            length(length), width(width), height(height) {}
    };

  /* Member functions */
  GraspTest(ros::NodeHandle& nh);
  void callback(const std_msgs::String msg);
  void executeCommand(std_msgs::String instruction);
  geometry_msgs::Pose createPose(float x, float y, float z, float roll,
    float pitch, float yaw);
  void moveGripper(double radium_mm, double angle_d, double palm_mm);
  void moveArm(float x, float y, float z, float roll, float pitch, float yaw);
  void moveRobot(double x, double y, double z, double roll, double pitch, double yaw,
    double radius_mm, double angle_d, double palm_mm);

  void setGripper(trajectory_msgs::JointTrajectory& posture,
    double radius_mm, double angle_d, double palm_mm);
  void pickObject(geometry_msgs::Pose objectCentre, double objectHeight,
    double objectDiameter);
  void placeObject(geometry_msgs::Pose dropPoint);

  moveit_msgs::CollisionObject makeBox(std::string id, std::string frame_id, Box box);
  void addCollsion();
  void disableCollisions(std::string name);

  /* Variables */
  ros::NodeHandle nh_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  moveit::planning_interface::MoveGroupInterface arm_group_{"panda_arm"};
  moveit::planning_interface::MoveGroupInterface hand_group_{"gripper"};
  moveit::planning_interface::MoveGroupInterface robot_group_{"all"};
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  geometry_msgs::Pose targetPose_;

  typedef Eigen::Matrix<double, 4, 1> Vector4d;

  std::string base_frame_ = "panda_link0";
  double min_distance_ = 0.2;
  double desired_distance_ = 0.25;

};