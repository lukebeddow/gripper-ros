#!/usr/bin/env python

import yaml
import rospkg

# instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# extract the path of the yaml file
control_path = rospack.get_path("gripper_control")
moveit_path = rospack.get_path("gripper_moveit")
description_path = rospack.get_path("gripper_description")
scripting_path = rospack.get_path("gripper_scripting")

# open the gripper information
with open(description_path + "//config//gripper.yaml") as file:
  gripper_details = yaml.safe_load(file)

# open the controller information for the arm and hand, then save
with open(control_path + "//config//control_arm_hand.yaml") as file:
  base_control_dict = yaml.safe_load(file)

# open the moveit controller list which includes the arm and hand
with open(moveit_path + "//config//controller_list_arm_hand.yaml") as file:
  base_moveit_dict = yaml.safe_load(file)

# exctract the details of the gripper configuration
is_segmented = gripper_details["is_segmented"]
num_segments = gripper_details["num_segments"]

# are we using a segmented finger? If so, we need to add extra joints
if is_segmented:

  # create lists of joints
  finger_1_joints = []
  finger_2_joints = []
  finger_3_joints = []
  template = "finger_{0}_segment_joint_{1}"
  for i in range(1, num_segments):
    finger_1_joints.append(template.format(1, i))
    finger_2_joints.append(template.format(2, i))
    finger_3_joints.append(template.format(3, i))

  # create segmented controller dictionary
  controller_name = "finger_{0}_controller"
  type_string = "segmented_controller/JointTrajectoryController"
  ns_string = "test"
  publish_rate = 50       
  control_dict = [{controller_name.format(1) :
                    {'type' : type_string,
                     'joints' : finger_1_joints,
                     'state_publisher_rate' : publish_rate}
                  },
                  {controller_name.format(2) :
                    {'type' : type_string,
                     'joints' : finger_2_joints,
                     'state_publisher_rate' : publish_rate}
                  },
                  {controller_name.format(3) :
                    {'type' : type_string,
                     'joints' : finger_3_joints,
                     'state_publisher_rate' : publish_rate}
                  }]

  # create controller_list dictionary
  type_string_2 = "FollowJointTrajectory"
  action_ns_string = "follow_joint_trajectory"
  moveit_dict = [{ "name" : controller_name.format(1),
                  "action_ns" : action_ns_string,
                  "type" : type_string_2,
                  "joints" : finger_1_joints},
                { "name" : controller_name.format(2),
                  "action_ns" : action_ns_string,
                  "type" : type_string_2,
                  "joints" : finger_2_joints},
                { "name" : controller_name.format(3),
                  "action_ns" : action_ns_string,
                  "type" : type_string_2,
                  "joints" : finger_3_joints}]

  # add the finger control information to the arm and hand controller
  for j in range(len(control_dict)):
    base_control_dict.update(control_dict[j])

  # add the finger control information to the controller_list
  for k in range(len(moveit_dict)):
    base_moveit_dict["controller_list"].append(moveit_dict[k])

# now we have a finished dictionary for each yaml file

# save as a new file which details the complete control information
with open(control_path + "//config//control_arm_hand_finger_autogenerated.yaml", "w") as file:
  yaml.dump(base_control_dict, file)

# save as a new file which details the complete controller list
with open(moveit_path + "//config//controller_list_arm_hand_finger_autogenerated.yaml", "w") as file:
  yaml.dump(base_moveit_dict, file)

# finally, save the details of our updates in the gripper.yaml file
gripper_details["previous_update"]["old_is_segmented"] = is_segmented
gripper_details["previous_update"]["old_num_segments"] = num_segments
with open(description_path + "//config//gripper.yaml", "w") as file:
  yaml.dump(gripper_details, file)

# end of script