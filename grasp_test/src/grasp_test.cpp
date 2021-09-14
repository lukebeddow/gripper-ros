#include <grasp_test.h>

GraspTest::GraspTest(ros::NodeHandle& nh)
{
  /* constructor */
  nh_ = nh;

  subscriber_ = nh_.subscribe<std_msgs::String>("/PluginTest/string", 10,
    &GraspTest::callback, this);

  // for testing
  addCollsion();

}

void GraspTest::callback(const std_msgs::String msg)
{
  /* callback function for when we receive a string message from a keypress */

  ROS_INFO_STREAM("grasp_test has received the letter " << msg << '\n');

  executeCommand(msg);
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
    moveRobot(0.2, 0.2, 0.8, 0.0, 0.0, 0.0, 140, 15.0, 0.0);
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
  // else if (msg.data == "8") {
  //   moveRobot(0, 0, 0, -0.7588, 0, 1, 0, 0.14, 0, 0);
  // }
  // else if (msg.data == "8") {
  //   moveRobot(0, 0, 0, -0.7588, 0, 1, 0, 0.14, 0, 0);
  // }
}

void GraspTest::moveGripper(double radius_mm, double angle_d, double palm_mm)
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
  double radius_m = radius_mm / 1000.0;
  double angle_r = (angle_d * 3.1415926535897 * -1) / 180.0;
  double palm_m = palm_mm / 1000.0;

  // set symmetric joint targets
  gripperJointTargets[0] = radius_m;
  gripperJointTargets[1] = angle_r;
  gripperJointTargets[2] = radius_m;
  gripperJointTargets[3] = angle_r;
  gripperJointTargets[4] = radius_m;
  gripperJointTargets[5] = angle_r;
  gripperJointTargets[6] = palm_m;

  hand_group_.setJointValueTarget(gripperJointTargets);

  // move the robot hand
  ROS_INFO("Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (hand_group_.plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  ROS_INFO("Visualising plan %s", success?"":"FAILED");

  hand_group_.move();

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

void GraspTest::moveArm(float x, float y, float z, float roll, float pitch, float yaw)
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
}

// void GraspTest::jointTarget(std::vector<double> jointTargets)
// {

// }

void GraspTest::moveRobot(double x, double y, double z, double roll, double pitch, double yaw,
  double radius_mm, double angle_d, double palm_mm)
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

  // to get the desired joint angles for the panda_arm, we create the pose
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
  arm_group_.setJointValueTarget(targetPose_);

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
  double radius_m = radius_mm / 1000.0;
  double angle_r = (angle_d * 3.1415926535897 * -1) / 180.0;
  double palm_m = palm_mm / 1000.0;

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

  // move the robot hand
  ROS_INFO("Attempting to plan the path");
  moveit::planning_interface::MoveGroupInterface::Plan my_plan;
  bool success = (robot_group_.plan(my_plan) ==
    moveit::planning_interface::MoveItErrorCode::SUCCESS);

  ROS_INFO("Visualising plan %s", success?"":"FAILED");

  robot_group_.move();

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
  double angle_d = 10;
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
