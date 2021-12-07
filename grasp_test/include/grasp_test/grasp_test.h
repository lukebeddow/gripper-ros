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

#include <gripper_msgs/pose.h>
#include <gripper_msgs/move_gripper.h>
#include <gripper_msgs/set_joints.h>
#include <gripper_msgs/set_pose.h>
#include <gripper_msgs/gripper_class.h>

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

    // functions
    gripper to_mm_deg() {
      return gripper(x * 1e3, th * (180.0 / 3.141592), z * 1e3);
    }
    gripper to_m() {
      return gripper(x * 1e-3, th, z * 1e-3);
    }
    
    // operator overloads
    gripper operator+(const gripper& a) {
      return gripper(a.x + x, a.th + th, a.z + z);
    }
    gripper operator-(const gripper& a) {
      return gripper(a.x - x, a.th - th, a.z - z);
    }
  };

  /* Member functions */
  GraspTest(ros::NodeHandle& nh);

  // subscriber callbacks
  void callback(const std_msgs::String msg);
  void virtual_msg_callback(const gripper_virtual_node::gripper_msg msg);
  void real_msg_callback(const gripper_virtual_node::gripper_msg msg);

  // service callbacks
  bool move_gripper_callback(gripper_msgs::move_gripper::Request &req,
    gripper_msgs::move_gripper::Response &res);
  bool set_joints_callback(gripper_msgs::set_joints::Request &req,
    gripper_msgs::set_joints::Response &res);
  bool set_pose_callback(gripper_msgs::set_pose::Request &req,
    gripper_msgs::set_pose::Response &res);

  // helper functions
  void executeCommand(std_msgs::String instruction);
  geometry_msgs::Pose createPose(float x, float y, float z, float roll,
    float pitch, float yaw);
  geometry_msgs::Pose getPose(std::string name);
  geometry_msgs::Quaternion rotateQuaternion(geometry_msgs::Quaternion msg1,
    geometry_msgs::Quaternion msg2);
  geometry_msgs::Quaternion vecToQuat(geometry_msgs::Vector3& vector);
  void waitForGripper();
  void waitUntilFinished();

  // move the gripper, the panda arm, or both
  bool moveGripper(double radium_mm, double angle_d, double palm_mm);
  bool moveGripper(Gripper gripper_pose);
  bool moveArm(float x, float y, float z, float roll, float pitch, float yaw);
  bool moveArm(geometry_msgs::Pose target_pose);
  bool moveRobot(double x, double y, double z, double roll, double pitch, double yaw,
    double radius_mm, double angle_d, double palm_mm);
  bool moveRobot(geometry_msgs::Pose desired_pose, Gripper gripper_pose);

  // functions using moveits pick and place built-in
  void setGripper(trajectory_msgs::JointTrajectory& posture,
    double radius_mm, double angle_d, double palm_mm);
  void pickObject(geometry_msgs::Pose objectCentre, double objectHeight,
    double objectDiameter);
  void placeObject(geometry_msgs::Pose dropPoint);

  // cartesian path planning
  void moveStraight(double distance);
  void moveStraight(double distance, geometry_msgs::Quaternion direction);
  void moveStraight(double distance, geometry_msgs::Vector3 direction);
  void moveStraight(double distance, geometry_msgs::Quaternion direction,
    bool is_global_direction);
  void moveStraight(double distance, geometry_msgs::Quaternion direction,
    geometry_msgs::Quaternion orientation);
  void moveStraight(double distance, geometry_msgs::Vector3 direction,
    geometry_msgs::Quaternion orientation);
  void moveStraight(double distance, geometry_msgs::Quaternion direction,
    bool is_global_direction, bool reorientate, geometry_msgs::Quaternion orientation);

  // functions for my pick and place
  bool myPick(geometry_msgs::Point grasp_point,
    geometry_msgs::Vector3 approach_vector);
  bool myPlace(geometry_msgs::Point place_point,
    geometry_msgs::Vector3 approach_vector);

  // collisions handling
  moveit_msgs::CollisionObject makeBox(std::string id, std::string frame_id, Box box);
  void addCollsion();
  void disableCollisions(std::string name);

  /* Variables */
  ros::NodeHandle nh_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;

  ros::ServiceServer move_gripper_srv_;
  ros::ServiceServer set_joints_srv_;
  ros::ServiceServer set_pose_srv_;

  moveit::planning_interface::MoveGroupInterface arm_group_{"panda_arm"};
  moveit::planning_interface::MoveGroupInterface hand_group_{"gripper"};
  moveit::planning_interface::MoveGroupInterface robot_group_{"all"};
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

  static constexpr double approach_distance_ = 0.2;   // should exceed gripper length
  static constexpr double offset_distance_ = 0.47;    // from panda_link8 to fingertip

  geometry_msgs::Pose targetPose_;

  Gripper gripper_default_{}; // initialise with default values
  Gripper gripper_virtual_;
  Gripper gripper_real_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener{tf_buffer_};

  typedef Eigen::Matrix<double, 4, 1> Vector4d;

  std::string base_frame_ = "panda_link0";
  double min_distance_ = 0.2;
  double desired_distance_ = 0.25;

  ros::Subscriber virtual_sub_;
  ros::Subscriber real_sub_;
  ros::Publisher gripper_pub_;

  gripper_virtual_node::gripper_state gripper_demand_msg_;
  gripper_virtual_node::gripper_msg gripper_virtual_status_;
  gripper_virtual_node::gripper_msg gripper_real_status_;

};