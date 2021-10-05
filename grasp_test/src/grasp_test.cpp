#include <grasp_test.h>

GraspTest::GraspTest(ros::NodeHandle& nh)
{
  /* constructor */
  nh_ = nh;

  subscriber_ = nh_.subscribe<std_msgs::String>("/PluginTest/string", 10,
    &GraspTest::callback, this);

  // tf2_ros::TransformListener tfListener(tfBuffer);

  // for testing
  addCollsion();

  gripper_pub_ = nh_.advertise<gripper_virtual_node::gripper_state>("/gripper/demand", 10);
  gripper_sub_ = nh_.subscribe<gripper_virtual_node::gripper_msg>("/gripper/status", 10,
    &GraspTest::gripper_msg_callback, this);

  gripper_default_.x = 140;
  gripper_default_.th = 0;
  gripper_default_.z = 0;

  ROS_INFO("Grasp test initialisation finished, ready to go");

}

void GraspTest::callback(const std_msgs::String msg)
{
  /* callback function for when we receive a string message from a keypress */

  ROS_INFO_STREAM("grasp_test has received the letter " << msg << '\n');

  executeCommand(msg);
}

void GraspTest::gripper_msg_callback(const gripper_virtual_node::gripper_msg msg)
{
  /* This function records the status of the gripper from the virtual node */

  gripper_status_msg_ = msg;
}

void GraspTest::executeCommand(const std_msgs::String msg)
{
  /* This function executes the given command */

  if (msg.data == "1") {
    moveArm(0.2, 0.2, 0.8, 0.0, 0.0, 0.0);
  }
  else if (msg.data == "2") {
    moveArm(-0.2, -0.2, 0.8, 0.0, 1.0, 1.0);
  }
  else if (msg.data == "3") {
    moveGripper(100.0, 0.0, 0.0);
  }
  else if (msg.data == "4") {
    moveGripper(130.0, 15.0, 100.0);
  }
  else if (msg.data == "5") {
    moveGripper(70.0, -15.0, 150.0);
  }
  else if (msg.data == "6") {
    geometry_msgs::Pose pose = createPose(0.2, 0.2, 0.8, 0.0, 0.0, 0.0);
    gripper gripper_pose{140, 15, 0};
    // moveRobot(0.2, 0.2, 0.8, 0.0, 0.0, 0.0, 140, 15.0, 0.0);
    moveRobot(pose, gripper_pose);
  }
  else if (msg.data == "7") {
    moveRobot(-0.2, -0.2, 0.8, 0.0, 1.0, 1.0, 70.0, -15.0, 150.0);
  }
  else if (msg.data == "8") {
    moveRobot(-0.4, 0.2, 0.7, 0.2, 1, 0.1, 70.0, -15.0, 150.0);
  }
  else if (msg.data == "9") {
    moveRobot(0.4, -0.2, 0.7, 0.2, 1.4, -2.1, 140, 15.0, 0.0);
  }
  else if (msg.data == "p") {
    geometry_msgs::Pose pose = createPose(0.2, 0.2, 0.4, 3.14159 / 2.0, 0, 0);
    pickObject(pose, 100, 100);
    placeObject(pose);
  }
  else if (msg.data == "c") {
    double distance = 0.1;
    geometry_msgs::Quaternion q;
    q.x = 1;
    q.y = 0;
    q.z = 0;
    q.w = 1;
    moveStraight(distance, q);
  }
  else if (msg.data == "m") {
    geometry_msgs::Pose pose = createPose(0.5, 0.0, 0.47, 3.14159 / 2.0, 0, 0);
    geometry_msgs::Vector3 v;
    v.x = 0;
    v.y = 0;
    v.z = -1;
    myPick(pose, v);
    gripper gripper_pose{100, 7, 70};
    moveGripper(gripper_pose);
  }
  // else if (msg.data == "8") {
  //   moveRobot(0, 0, 0, -0.7588, 0, 1, 0, 0.14, 0, 0);
  // }
  // else if (msg.data == "8") {
  //   moveRobot(0, 0, 0, -0.7588, 0, 1, 0, 0.14, 0, 0);
  // }
}

geometry_msgs::Pose GraspTest::createPose(float x, float y, float z, float roll,
  float pitch, float yaw)
{
  /* This function creates a pose from the given inputs */

  geometry_msgs::Pose pose;

  // setup the goal position
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  // setup the goal orientation
  tf2::Quaternion q;
  geometry_msgs::Quaternion q_msg;
  q.setRPY(roll, pitch, yaw);
  q.normalize();
  q_msg = tf2::toMsg(q);
  pose.orientation = q_msg;

  return pose;
}

bool GraspTest::moveGripper(double radius_mm, double angle_d, double palm_mm)
{
  /* overload function with basic inputs */

  gripper gripper_pose;
  gripper_pose.x = radius_mm;
  gripper_pose.th = angle_d;
  gripper_pose.z = palm_mm;

  return moveGripper(gripper_pose);
}

bool GraspTest::moveGripper(gripper gripper_pose)
{
  /* Set joint targets for the gripper, the order is vital and must be:
      [0] finger_1_prismatic_joint
      [1] finger_1_revolute_joint
      [2] finger_2_prismatic_joint
      [3] finger_2_revolute_joint
      [4] finger_3_prismatic_joint
      [5] finger_3_revolute_joint
      [6] palm_prismatic_joint
     Note that a negative angle_d means fingers tilted outwards, positive inwards.
  */

  std::vector<double> gripperJointTargets;
  gripperJointTargets.resize(7);

  // convert units
  double radius_m = gripper_pose.x / 1000.0;
  double angle_r = (gripper_pose.th * 3.1415926535897 * -1) / 180.0;
  double palm_m = gripper_pose.z / 1000.0;

  // set symmetric joint targets
  gripperJointTargets[0] = radius_m;
  gripperJointTargets[1] = angle_r;
  gripperJointTargets[2] = radius_m;
  gripperJointTargets[3] = angle_r;
  gripperJointTargets[4] = radius_m;
  gripperJointTargets[5] = angle_r;
  gripperJointTargets[6] = palm_m;

  hand_group_.setJointValueTarget(gripperJointTargets);

  // publish the gripper demand
  gripper_demand_msg_.x = radius_m;
  gripper_demand_msg_.th = angle_r;
  gripper_demand_msg_.z = palm_m;
  gripper_pub_.publish(gripper_demand_msg_);

  // move the robot hand
  ROS_INFO("Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (hand_group_.plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  ROS_INFO("Visualising plan %s", success?"":"FAILED");

  hand_group_.move();

  return success;
}

bool GraspTest::moveArm(float x, float y, float z, float roll, float pitch, float yaw)
{
  /* This function moves the move_group to the target position */

  // setup the goal position
  targetPose_.position.x = x;
  targetPose_.position.y = y;
  targetPose_.position.z = z;

  // setup the goal orientation
  tf2::Quaternion q;
  geometry_msgs::Quaternion q_msg;
  q.setRPY(roll, pitch, yaw);
  q.normalize();
  q_msg = tf2::toMsg(q);
  targetPose_.orientation = q_msg;

  // setup the target pose
  ROS_INFO("Setting pose target");
  arm_group_.setPoseTarget(targetPose_);

  // move_group.setEndEffectorLink("panda_link8");
  // move_group.setEndEffector("gripper");

  /* For some reason the below leads to different positions even with the same inputs */
  // arm_group_.setPositionTarget(x, y, z, "panda_link8");
  // arm_group_.setRPYTarget(roll, pitch, yaw, "panda_link8");

  // move the robot hand
  ROS_INFO("Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (arm_group_.plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  ROS_INFO("Visualising plan %s", success?"":"FAILED");

  arm_group_.move();

  return success;
}

// void GraspTest::jointTarget(std::vector<double> jointTargets)
// {

// }

bool GraspTest::moveRobot(double x, double y, double z, double roll, double pitch, double yaw,
  double radius_mm, double angle_d, double palm_mm)
{
  /* Function overload with more generic inputs */

  geometry_msgs::Pose desired_pose;
  gripper gripper_pose;

  // setup the goal position
  desired_pose.position.x = x;
  desired_pose.position.y = y;
  desired_pose.position.z = z;

  // setup the goal orientation
  tf2::Quaternion q;
  geometry_msgs::Quaternion q_msg;
  q.setRPY(roll, pitch, yaw);
  q.normalize();
  q_msg = tf2::toMsg(q);
  desired_pose.orientation = q_msg;

  // setup the goal gripper pose
  gripper_pose.x = radius_mm;
  gripper_pose.th = angle_d;
  gripper_pose.z = palm_mm;

  return moveRobot(desired_pose, gripper_pose);
}

bool GraspTest::moveRobot(geometry_msgs::Pose desired_pose, gripper gripper_pose)
{
  /* This function attempts to move the entire robot at the same time.
     Joint names in order are:
     panda_joint1
     panda_joint2
     panda_joint3
     panda_joint4
     panda_joint5
     panda_joint6
     panda_joint7
     finger_1_prismatic_joint
     finger_1_revolute_joint
     finger_2_prismatic_joint
     finger_2_revolute_joint
     finger_3_prismatic_joint
     finger_3_revolute_joint
     palm_prismatic_joint
     */

  // std::vector<std::string> names = robot_group_.getJointNames();
  // std::cout << "Joint names in order are:\n";
  // for (int i = 0; i < names.size(); i++) {
  //   std::cout << names[i] << '\n';
  // }

  std::vector<double> jointTargets;
  jointTargets.resize(14);

  // setup the target pose
  arm_group_.setJointValueTarget(desired_pose);

  // now extract the joint values
  robot_state::RobotState state = arm_group_.getJointValueTarget();

  // extract the joint positions from the pointers returned by the member fcn
  jointTargets[0] = *state.getJointPositions("panda_joint1");
  jointTargets[1] = *state.getJointPositions("panda_joint2");
  jointTargets[2] = *state.getJointPositions("panda_joint3");
  jointTargets[3] = *state.getJointPositions("panda_joint4");
  jointTargets[4] = *state.getJointPositions("panda_joint5");
  jointTargets[5] = *state.getJointPositions("panda_joint6");
  jointTargets[6] = *state.getJointPositions("panda_joint7");

  // convert units
  double radius_m = gripper_pose.x / 1000.0;
  double angle_r = (gripper_pose.th * 3.1415926535897 * -1) / 180.0;
  double palm_m = gripper_pose.z / 1000.0;

  // set symmetric joint targets
  jointTargets[7] = radius_m;
  jointTargets[8] = angle_r;
  jointTargets[9] = radius_m;
  jointTargets[10] = angle_r;
  jointTargets[11] = radius_m;
  jointTargets[12] = angle_r;
  jointTargets[13] = palm_m;

  ROS_INFO("Setting joint target");
  robot_group_.setJointValueTarget(jointTargets);

  // publish the gripper demand
  gripper_demand_msg_.x = radius_m;
  gripper_demand_msg_.th = angle_r;
  gripper_demand_msg_.z = palm_m;
  gripper_pub_.publish(gripper_demand_msg_);

  // move the robot hand
  ROS_INFO("Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (robot_group_.plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  ROS_INFO("Visualising plan %s", success?"":"FAILED");

  robot_group_.move();

  return success;
}

void GraspTest::setGripper(trajectory_msgs::JointTrajectory& posture,
  double radius_mm, double angle_d, double palm_mm)
{
  /* This function sets the posture message for the gripper at the positions
     specified */

  // add the gripper joints
  posture.joint_names.resize(7);
  posture.joint_names[0] = "finger_1_prismatic_joint";
  posture.joint_names[1] = "finger_1_revolute_joint";
  posture.joint_names[2] = "finger_2_prismatic_joint";
  posture.joint_names[3] = "finger_2_revolute_joint";
  posture.joint_names[4] = "finger_3_prismatic_joint";
  posture.joint_names[5] = "finger_3_revolute_joint";
  posture.joint_names[6] = "palm_prismatic_joint";

  // convert units
  double radius_m = radius_mm / 1000.0;
  double angle_r = (angle_d * 3.1415926535897 * -1) / 180.0;
  double palm_m = palm_mm / 1000.0;

  // set the positions
  posture.points.resize(1);
  posture.points[0].positions.resize(7);
  posture.points[0].positions[0] = radius_m;
  posture.points[0].positions[1] = angle_r;
  posture.points[0].positions[2] = radius_m;
  posture.points[0].positions[3] = angle_r;
  posture.points[0].positions[4] = radius_m;
  posture.points[0].positions[5] = angle_r;
  posture.points[0].positions[6] = palm_m;

  // time to wait for grasp posture to be reached
  posture.points[0].time_from_start = ros::Duration(2.0);

}

void GraspTest::pickObject(geometry_msgs::Pose objectCentre, double objectHeight,
  double objectDiameter)
{
  /* This function uses moveit grasping to grasp an object */

  // create a one element vector to hold the grasp posture
  std::vector<moveit_msgs::Grasp> grasps(1);

  // define grasp approach/retreat as from above
  Vector4d pre_grasp_vector(0, 0, -1, 1);
  Vector4d post_grasp_vector(0, 0, 1, 1);

  // configure the grasping message
  grasps[0].grasp_pose.header.frame_id = base_frame_;

  double z = objectCentre.position.z - objectHeight / 2 + 0.5;

  // // setup the goal position
  // grasps[0].grasp_pose.pose.position.x = objectCentre.position.x;
  // grasps[0].grasp_pose.pose.position.y = objectCentre.position.y;
  // grasps[0].grasp_pose.pose.position.z = z;

  // // setup the goal orientation
  // grasps[0].grasp_pose.pose.orientation = objectCentre.orientation;

  // FOR TESTING! HARDCODE POSITION AND ORIENTATION
  // setup the goal position
  grasps[0].grasp_pose.pose.position.x = 0.5;
  grasps[0].grasp_pose.pose.position.y = 0;
  grasps[0].grasp_pose.pose.position.z = 0.465; //0.47
  grasps[0].grasp_pose.pose.orientation.x = 1;
  grasps[0].grasp_pose.pose.orientation.y = 0;
  grasps[0].grasp_pose.pose.orientation.z = 0;
  grasps[0].grasp_pose.pose.orientation.w = 0;

  // configure pre-grasp
  grasps[0].pre_grasp_approach.direction.header.frame_id = base_frame_;
  grasps[0].pre_grasp_approach.direction.vector.x = pre_grasp_vector(0);
  grasps[0].pre_grasp_approach.direction.vector.y = pre_grasp_vector(1);
  grasps[0].pre_grasp_approach.direction.vector.z = pre_grasp_vector(2);
  grasps[0].pre_grasp_approach.min_distance = min_distance_;
  grasps[0].pre_grasp_approach.desired_distance = desired_distance_;

  // configure the post-grasp
  grasps[0].post_grasp_retreat.direction.header.frame_id = base_frame_;
  grasps[0].post_grasp_retreat.direction.vector.x = post_grasp_vector(0);
  grasps[0].post_grasp_retreat.direction.vector.y = post_grasp_vector(1);
  grasps[0].post_grasp_retreat.direction.vector.z = post_grasp_vector(2);
  grasps[0].post_grasp_retreat.min_distance = min_distance_;
  grasps[0].post_grasp_retreat.desired_distance = desired_distance_;

  // set the gripper configuration (110, 8, 30 works for my test grasp)
  double radius_mm = 110;
  double angle_d = 9;
  double palm_mm = 30;

  setGripper(grasps[0].pre_grasp_posture, 140, 0, 0);
  setGripper(grasps[0].grasp_posture, radius_mm, angle_d, palm_mm);

  ROS_INFO("Trying pick operation now");

  arm_group_.pick("object", grasps);
  // robot_group_.pick("object", grasps);

}

void GraspTest::placeObject(geometry_msgs::Pose dropPoint)
{
  /* This function uses moveit grasping to grasp an object */

  // create a one element vector to hold the placement posture
  std::vector<moveit_msgs::PlaceLocation> places(1);

  // define grasp approach/retreat as from above
  Vector4d pre_place_vector(0, 0, -1, 1);
  Vector4d post_place_vector(0, 0, 1, 1);

  // configure the grasping message
  places[0].place_pose.header.frame_id = base_frame_;

  // // setup the goal position
  // places[0].place_pose.pose.position.x = objectCentre.position.x;
  // places[0].place_pose.pose.position.y = objectCentre.position.y;
  // places[0].place_pose.pose.position.z = z;

  // // setup the goal orientation
  // places[0].place_pose.pose.orientation = objectCentre.orientation;

  // FOR TESTING! HARDCODE POSITION AND ORIENTATION
  // setup the goal position
  places[0].place_pose.pose.position.x = 0.5;
  places[0].place_pose.pose.position.y = 0;
  places[0].place_pose.pose.position.z = 0.2; //0.47
  places[0].place_pose.pose.orientation.x = 0;
  places[0].place_pose.pose.orientation.y = 0;
  places[0].place_pose.pose.orientation.z = 0;
  places[0].place_pose.pose.orientation.w = 1;

  // configure pre-grasp
  places[0].pre_place_approach.direction.vector.x = pre_place_vector(0);
  places[0].pre_place_approach.direction.vector.y = pre_place_vector(1);
  places[0].pre_place_approach.direction.vector.z = pre_place_vector(2);
  places[0].pre_place_approach.min_distance = min_distance_ / 2;
  places[0].pre_place_approach.desired_distance = desired_distance_ / 2;

  // configure the post-grasp
  places[0].post_place_retreat.direction.header.frame_id = base_frame_;
  places[0].post_place_retreat.direction.vector.x = post_place_vector(0);
  places[0].post_place_retreat.direction.vector.y = post_place_vector(1);
  places[0].post_place_retreat.direction.vector.z = post_place_vector(2);
  places[0].post_place_retreat.min_distance = min_distance_ / 2;
  places[0].post_place_retreat.desired_distance = desired_distance_ / 2;

  // set the gripper configuration
  double radius_mm = 110;
  double angle_d = -10;
  double palm_mm = 0;

  setGripper(places[0].post_place_posture, radius_mm, angle_d, palm_mm);

  ROS_INFO("Trying place operation now");

  arm_group_.place("object", places);
  // robot_group_.pick("object", grasps);

}

moveit_msgs::CollisionObject GraspTest::makeBox(std::string id, std::string frame_id, Box box)
{
  // Makes a Box collision object at given location with given dimensions. 

  moveit_msgs::CollisionObject collision_object;
  
  // input header information
  collision_object.id = id;
  collision_object.header.frame_id = frame_id;

  /* Define the primitive and its dimensions. */
  collision_object.primitives.resize(1);
  collision_object.primitives[0].type = collision_object.primitives[0].BOX;
  collision_object.primitives[0].dimensions.resize(3);
  collision_object.primitives[0].dimensions[0] = box.length;
  collision_object.primitives[0].dimensions[1] = box.width;
  collision_object.primitives[0].dimensions[2] = box.height;

  // // determine orientation
  // tf::Quaternion q;
  // geometry_msgs::Quaternion q_msg;
  // q.setRPY(box.roll, box.pitch, box.yaw);
  // tf::quaternionTFToMsg (q, q_msg);
  
  /* Define the pose of the table: center of the cube. */
  collision_object.primitive_poses.resize(1);
  collision_object.primitive_poses[0].position.x =  box.x;
  collision_object.primitive_poses[0].position.y =  box.y;
  collision_object.primitive_poses[0].position.z =  box.z;
  // collision_object.primitive_poses[0].orientation = q_msg;
  collision_object.primitive_poses[0].orientation.x = 0;
  collision_object.primitive_poses[0].orientation.y = 0;
  collision_object.primitive_poses[0].orientation.z = 0;
  collision_object.primitive_poses[0].orientation.w = 1;

  collision_object.operation = collision_object.ADD;

  return collision_object;
}

void GraspTest::addCollsion()
{
  // Creating Environment
  // ^^^^^^^^^^^^^^^^^^^^
  // Create vector to hold collision objects.
  std::vector<moveit_msgs::CollisionObject> collision_objects;

  // define object size
  double object_height = 0.2;
  double object_length = 0.02;
  double object_width = 0.02;

  // create the object
  Box object_box {0.5, 0.3, object_height / 2,
                  0, 0, 0,
                  object_length, object_width, object_height};

  collision_objects.push_back(makeBox("object", "panda_link0", object_box));

  // create a floor
  Box floor{0, 0, -0.11,
            0, 0, 0,
            10, 10, 0.1};
  collision_objects.push_back(makeBox("floor", "panda_link0", floor));

  planning_scene_interface.applyCollisionObjects(collision_objects);

  /* Now, disable collision checking on the object, since we care only about Gazebo
     and in fact we intend for there to be collisions */
  disableCollisions("object");

}

void GraspTest::disableCollisions(std::string name)
{
  /* Disable all collision checking for a specified object */

  // create the planning scene monitor
  robot_model_loader::RobotModelLoaderPtr robot_model_loader
    (new robot_model_loader::RobotModelLoader("robot_description"));
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor
    (new planning_scene_monitor::PlanningSceneMonitor(robot_model_loader));

  // start monitoring our scene
  planning_scene_monitor->startSceneMonitor();
  planning_scene_monitor->startWorldGeometryMonitor();
  planning_scene_monitor->startStateMonitor();

  // get a safe pointer to the planning scene, and extract the allowed collision matrix
  planning_scene_monitor::LockedPlanningSceneRW ls(planning_scene_monitor);
  collision_detection::AllowedCollisionMatrix& acm = ls->getAllowedCollisionMatrixNonConst();

  // set that all collisions are allowed for the object
  acm.setEntry(name, true);

  // std::cout << "\nAllowedCollisionMatrix:\n";
  // acm.print(std::cout);

  // create a new planning scene message
  moveit_msgs::PlanningScene updated_scene;
  ls->getPlanningSceneDiffMsg(updated_scene);
  updated_scene.robot_state.is_diff = true;
  // ls->setPlanningSceneDiffMsg(updated_scene); //needed? No

  // update to the new planning scene
  planning_scene_interface.applyPlanningScene(updated_scene);
}

geometry_msgs::Quaternion GraspTest::rotateQuaternion(geometry_msgs::Quaternion msg1,
  geometry_msgs::Quaternion msg2)
{
  /* This function rotates a quaternion message by another quaternion */

  // convert quaternion messages into tf2 quaternions
  tf2::Quaternion q1;
  tf2::Quaternion q2;
  tf2::fromMsg(msg1, q1);
  tf2::fromMsg(msg2, q2);

  // apply the rotation
  q1 *= q2;

  return tf2::toMsg(q1);
}

void GraspTest::moveStraight(double distance, geometry_msgs::Quaternion q)
{
  /* This function moves the end effector in a straight line in the given direction */

  /* see: https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_interface/src/move_group_interface_tutorial.cpp
          http://docs.ros.org/en/jade/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroup.html#a1ffa321c3085f6be8769a892dbc89b30
          http://docs.ros.org/en/melodic/api/tf2_geometry_msgs/html/c++/namespacetf2.html#a82ca47c6f5b0360e6c5b250dca719a78
  */

  // start pose is not required, but helps with visualisation if added as a waypoint
  geometry_msgs::Pose start_pose;
  geometry_msgs::Pose destination_pose;
  geometry_msgs::TransformStamped start_tf;

  try {
    start_tf = tf_buffer_.lookupTransform("panda_link0", "panda_link8", ros::Time(0));
  }
  catch (tf2::TransformException &ex) {
    ROS_WARN("%s", ex.what());
    ros::Duration(1.0).sleep();
  }

  // save transform data in pose message
  start_pose.position.x = start_tf.transform.translation.x;
  start_pose.position.y = start_tf.transform.translation.y;
  start_pose.position.z = start_tf.transform.translation.z;
  start_pose.orientation = start_tf.transform.rotation;

  // create vector to hold waypoints
  std::vector<geometry_msgs::Pose> waypoints;
  // waypoints.push_back(start_pose);

  // convert quaternion messages into tf2 quaternions
  tf2::Quaternion start_q;
  tf2::Quaternion rotate_q;
  tf2::fromMsg(start_pose.orientation, start_q);
  tf2::fromMsg(q, rotate_q);

  // apply the rotation
  start_q *= rotate_q;

  // convert the final rotation back to a quaternion message
  q = tf2::toMsg(start_q);

  // create two vectors to be rotated
  geometry_msgs::Point vectorA;
  geometry_msgs::Point vectorB;

  // [0, 0, distance]
  vectorA.z = distance;

  // create transform to adjust start pose with, and apply rotation
  geometry_msgs::TransformStamped adjust_tf;
  adjust_tf.transform.translation.x = 0;
  adjust_tf.transform.translation.y = 0;
  adjust_tf.transform.translation.z = 0;
  // adjust_tf.transform.rotation = q;
  adjust_tf.transform.rotation = rotateQuaternion(start_pose.orientation, q);

  // apply the rotation
  tf2::doTransform(vectorA, vectorB, adjust_tf);

  // input the result into the destination pose
  destination_pose.position.x = start_pose.position.x + vectorB.x;
  destination_pose.position.y = start_pose.position.y + vectorB.y;
  destination_pose.position.z = start_pose.position.z + vectorB.z;
  destination_pose.orientation = start_pose.orientation;

  // insert target pose
  waypoints.push_back(destination_pose);

  // create a trajectory message to hold the computed trajectory
  moveit_msgs::RobotTrajectory trajectory;

  // define parameters
  double jump_threshold = 0.0; // nb 0.0 disables, dangerous on real hardware
  double eef_step = 0.001;     // could do distance / 10 since 10 is min steps

  // computeCartesianPath requires at least 10 steps
  if (eef_step > distance / 10) {
    eef_step = distance / 10;
  }
  
  // compute the trajectory
  double fraction = arm_group_.computeCartesianPath(waypoints, eef_step,
    jump_threshold, trajectory);

  ROS_INFO("Visualizing plan 4 (Cartesian path) (%.2f%% acheived)", fraction * 100.0);

  // // visualise the path in RViz
  // visual_tools.deleteAllMarkers();
  // visual_tools.publishText(text_pose, "Cartesian Path", rvt::WHITE, rvt::XLARGE);
  // visual_tools.publishPath(waypoints, rvt::LIME_GREEN, rvt::SMALL);
  // for (std::size_t i = 0; i < waypoints.size(); ++i)
  //   visual_tools.publishAxisLabeled(waypoints[i], "pt" + std::to_string(i), rvt::SMALL);
  // visual_tools.trigger();
  // visual_tools.prompt("Press 'next' in the RvizVisualToolsGui window to continue the demo");

  // execute the trajecotry
  arm_group_.execute(trajectory);
  // move_group_interface.execute(trajectory)
}

void GraspTest::waitForGripper()
{
  /* This function is blocking and waits until the gripper has satisfied the
  demand pose */

  ROS_INFO("Entering the waitForGripper function");

  ros::Duration(0.1).sleep();

  while (!gripper_status_msg_.is_target_reached) {
    // ros::spinOnce();
    ros::Duration(0.01).sleep();
  }

  ROS_INFO("Leaving the waitForGripper function");
}

bool GraspTest::myPick(geometry_msgs::Pose grasp_pose,
  geometry_msgs::Vector3 approach_vector)
{
  /* This function attempts to perform a pick action */

  /* The pick action consists of several steps:
      1. Move to pre-pick point
      2. Configure gripper correctly
      3. Move along approach vector to object
      4. Close the gripper
      5. Feedback loop to adjust gripper and position to secure object
      6. Move backwards along approach vector
      7. Complete
  */

  // construct from eigen
  Eigen::Vector3d A(0, 0, 1);
  Eigen::Vector3d B(approach_vector.x, approach_vector.y, approach_vector.z);
  B = B.normalized();
  Eigen::Quaterniond q_eigen;
  q_eigen.setFromTwoVectors(A, B);
  geometry_msgs::Quaternion q_fromEigen = tf2::toMsg(q_eigen);
  ROS_INFO_STREAM("q_eigen is: " << q_fromEigen);

  // construct from axis and angle
  tf2Scalar angle(0.0);
  // tf2::Vector3 axis(approach_vector.x, approach_vector.y, approach_vector.z);
  tf2::Vector3 axis(0.0, 0.0, -1.0);
  axis.normalize();

  tf2Scalar qx = 0;
  tf2Scalar qy = 0;
  tf2Scalar qz = -0.7071068;
  tf2Scalar qw = -0.7071068;
  tf2::Quaternion q_ninety(qx, qy, qz, qw);

  tf2::Quaternion q_test(axis, angle);
  q_test *= q_ninety;

  geometry_msgs::Quaternion q_print = tf2::toMsg(q_test);
  ROS_INFO_STREAM("q_test is: " << q_print);
  
  // define the approach distance before grasp (needs to exceed gripper finger length)
  double approach_distance = 0.2;

  // define the closed state of the gripper
  gripper closed_gripper;
  closed_gripper.x = 120;
  closed_gripper.th = 12;
  closed_gripper.z = 20;

  // define a quaternion that flips another quaternion 180deg about y axis
  geometry_msgs::Quaternion q_backwards;
  q_backwards.x = 0;
  q_backwards.y = 1;
  q_backwards.z = 0;
  q_backwards.w = 0;

  // define the pre-pick pose
  geometry_msgs::Pose pre_pick_pose;
  pre_pick_pose.position.x = grasp_pose.position.x - B.x() * approach_distance;
  pre_pick_pose.position.y = grasp_pose.position.y - B.y() * approach_distance;
  pre_pick_pose.position.z = grasp_pose.position.z - B.z() * approach_distance;
  pre_pick_pose.orientation = q_fromEigen;

  ROS_INFO_STREAM("pre_pick_pose is " << pre_pick_pose.position.x
    << ", " << pre_pick_pose.position.y << ", " << pre_pick_pose.position.z);
  ROS_INFO_STREAM("grasp_pose is " << grasp_pose.position.x
    << ", " << grasp_pose.position.y << ", " << grasp_pose.position.z);

  bool succeeded = false;

  /* 1. Move to pre-pick point */
  succeeded = moveRobot(pre_pick_pose, gripper_default_);

  if (!succeeded) return false;

  waitForGripper();

  ROS_INFO("Robot moved to pre-pick pose");

  /* 2. Configure the gripper correctly */
  // moveRobot does not return until gripper reaches desired positions

  /* 3. Move along approach vector to object */
  moveStraight(approach_distance, pre_pick_pose.orientation);

  ROS_INFO("Robot moved to grasping pose");

  /* 4. Close the gripper */
  succeeded = moveGripper(closed_gripper);

  if (!succeeded) return false;

  waitForGripper();

  /* 5. Feedback loop */
  // not currently implemented

  /* 6. Move backwards along approach vector */
  moveStraight(approach_distance, rotateQuaternion(pre_pick_pose.orientation, q_backwards));

  /* 7. Complete */
  // check if object is still in grasp?

}
