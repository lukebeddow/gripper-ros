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

# create lists of joints
num_segments = 10
finger_1_joints = []
finger_2_joints = []
finger_3_joints = []
template = "finger_{0}_segment_joint{1}"
for i in range(1, num_segments):
  finger_1_joints.append(template.format(1, i))
  finger_2_joints.append(template.format(2, i))
  finger_3_joints.append(template.format(3, i))

# create segmented controller dictionary
controller_name = "finger_{0}_controller"
type_string = "segmented_controller/JointTrajectoryController"
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
                 "joint_names" : finger_1_joints},
               { "name" : controller_name.format(2),
                 "action_ns" : action_ns_string,
                 "type" : type_string_2,
                 "joint_names" : finger_2_joints},
               { "name" : controller_name.format(3),
                 "action_ns" : action_ns_string,
                 "type" : type_string_2,
                 "joint_names" : finger_3_joints}]

with open(control_path + "//config//panda_control.yaml") as f:
  base_control_dict = yaml.safe_load(f)

for j in range(len(control_dict)):
  base_control_dict.update(control_dict[j])

with open(control_path + "//config//auto_yaml.yaml", "w") as file:
  yaml.dump(base_control_dict, file)

with open(moveit_path + "//config//controllers_panda.yaml") as f:
  base_moveit_dict = yaml.safe_load(f)

for k in range(len(moveit_dict)):
  base_moveit_dict["controller_list"].append(moveit_dict[k])

with open(moveit_path + "//config//auto_yaml.yaml", "w") as file:
  yaml.dump(base_moveit_dict, file)