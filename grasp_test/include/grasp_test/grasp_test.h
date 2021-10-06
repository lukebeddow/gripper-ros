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
#include <tf2/LinearMath/Scalar.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf/transform_listener.h>

#include <gripper_virtual_node/gripper_state.h>
#include <gripper_virtual_node/gripper_msg.h>

// #include <moveit_msgs/GetPlanningScene.h>
// #include <moveit/robot_model_loader/robot_model_loader.h>
// #include <moveit/kinematic_constraints/utils.h>
// #include <moveit/planning_scene/planning_scene.h>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>

class GraspTest
{
public:

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

  struct gripper {
    double x;
    double th;
    double z;

    // constructor
    gripper() {}
    gripper(double x, double th, double z)
    : x(x), th(th), z(z) {}
  };

  /* Member functions */
  GraspTest(ros::NodeHandle& nh);
  void callback(const std_msgs::String msg);
  void gripper_msg_callback(const gripper_virtual_node::gripper_msg msg);
  void executeCommand(std_msgs::String instruction);
  geometry_msgs::Pose createPose(float x, float y, float z, float roll,
    float pitch, float yaw);
  geometry_msgs::Quaternion rotateQuaternion(geometry_msgs::Quaternion msg1,
    geometry_msgs::Quaternion msg2);
  geometry_msgs::Quaternion vecToQuat(geometry_msgs::Vector3& vector);
  bool moveGripper(double radium_mm, double angle_d, double palm_mm);
  bool moveGripper(gripper gripper_pose);
  bool moveArm(float x, float y, float z, float roll, float pitch, float yaw);
  bool moveArm(geometry_msgs::Pose target_pose);
  bool moveRobot(double x, double y, double z, double roll, double pitch, double yaw,
    double radius_mm, double angle_d, double palm_mm);
  bool moveRobot(geometry_msgs::Pose desired_pose, gripper gripper_pose);

  void setGripper(trajectory_msgs::JointTrajectory& posture,
    double radius_mm, double angle_d, double palm_mm);
  void pickObject(geometry_msgs::Pose objectCentre, double objectHeight,
    double objectDiameter);
  void placeObject(geometry_msgs::Pose dropPoint);
  void moveStraight(double distance);
  void moveStraight(double distance, geometry_msgs::Quaternion direction);
  void moveStraight(double distance, geometry_msgs::Quaternion direction,
    bool is_global_direction);
  bool myPick(geometry_msgs::Point grasp_point,
    geometry_msgs::Vector3 approach_vector);
  bool myPlace(geometry_msgs::Point place_point,
    geometry_msgs::Vector3 approach_vector);

  void waitForGripper();

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

  static constexpr double approach_distance_ = 0.2;   // should exceed gripper length
  static constexpr double offset_distance_ = 0.47;    // from panda_link8 to fingertip

  geometry_msgs::Pose targetPose_;

  gripper gripper_default_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener{tf_buffer_};

  typedef Eigen::Matrix<double, 4, 1> Vector4d;

  std::string base_frame_ = "panda_link0";
  double min_distance_ = 0.2;
  double desired_distance_ = 0.25;

  ros::Subscriber gripper_sub_;
  ros::Publisher gripper_pub_;

  gripper_virtual_node::gripper_state gripper_demand_msg_;
  gripper_virtual_node::gripper_msg gripper_status_msg_;

};