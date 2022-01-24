#!/usr/bin/env python

import yaml
import rospkg 
import lxml.etree as etree
import copy
import numpy as np
from math import floor, ceil

# open the grippr details yaml file to extract parameters
rospack = rospkg.RosPack()
description_path = rospack.get_path("gripper_description")
with open(description_path + "/config/gripper.yaml") as file:
  gripper_details = yaml.safe_load(file)


# ----- essential user defined parameters ----- #

# exctract the details of the gripper configuration from yaml file
is_segmented = gripper_details["gripper_config"]["is_segmented"]
num_segments = gripper_details["gripper_config"]["num_segments"]

# starting configuration of the robot joints
j = {
  "panda_joint1": 0.01,
  "panda_joint2": 0.02,
  "panda_joint3": 0.03,
  "panda_joint4": 0.04,
  "panda_joint5": 0.05,
  "panda_joint6": 1.0,
  "panda_joint7": 0.07,
  "gripper_prismatic": 140e-3,
  "gripper_revolute": 0.0,
  "gripper_palm": 0.0
}

# panda parameters
panda_control = "motor"

# gripper parameters
gripper_control = "motor"
force_limit_prismatic = 10.0
force_limit_revolute = 10.0
force_limit_palm = 10.0

# finger dummy parameters
finger_control = "motor"
finger_joint_stiffness = 5 # 10 appears more realistic

# task parameters
num_object_freejoints = 12

# ----- end essential user defined parameters ----- #


# setup the objects to the side in a grid formation
grid_xrange = [-1, 1]
grid_ystart = 1
spacing = 0.2
max_objects = 100

per_x = int(floor((grid_xrange[1] - grid_xrange[0]) / float(spacing)))
num_y = int(ceil(max_objects / float(per_x)))
object_X = [(grid_xrange[0] + (spacing / 2.) + spacing * i) for i in range(per_x)] * num_y
object_Y = [(grid_ystart + spacing * j) for j in range(num_y) for i in range(per_x)]

object_Z = 0.1
object_Q = "0 0 0 1"
object_qpos = ""
for i in range(num_object_freejoints):
  object_qpos += " " + str(object_X[i]) + " " + str(object_Y[i]) + " " + str(object_Z)
  object_qpos += " " + object_Q

# now automatically generate xml strings to encode these starting values
if is_segmented:
  finger_joint_qpos = "0 " * (num_segments - 1)
else:
  finger_joint_qpos = ""

gripper_qpos = "{0} {1} {2} {0} {1} {2} {0} {1} {2} {3}".format(
  j["gripper_prismatic"], j["gripper_revolute"], finger_joint_qpos, j["gripper_palm"]
)
panda_qpos = "{0} {1} {2} {3} {4} {5} {6}".format(
  j["panda_joint1"], j["panda_joint2"], j["panda_joint3"], j["panda_joint4"],
  j["panda_joint5"], j["panda_joint6"], j["panda_joint7"]
)
# freejoint_qpos = "0 " * (num_object_freejoints * 7 - 1) + "0"

# define the actuated gripper and panda joints
gripper_joints = [
  "finger_1_prismatic_joint", "finger_1_revolute_joint",
  "finger_2_prismatic_joint", "finger_2_revolute_joint",
  "finger_3_prismatic_joint", "finger_3_revolute_joint",
  "palm_prismatic_joint"]
panda_joints = ["panda_joint{0}".format(i) for i in range(1,8)]
finger_joints = ["finger_{0}_segment_joint_{1}".format(i, j) for i in range(1,4) 
                  for j in range(1, num_segments)]


gripper_keyframe = """ 
  <keyframe>
    <key name="initial pose"
         time="0"
         qpos="{0}"
    />
  </keyframe>
""".format(gripper_qpos)

panda_keyframe = """
  <keyframe>
    <key name="initial pose"
         time="0"
         qpos="{0}"
    />
  </keyframe>
""".format(panda_qpos)

panda_and_gripper_keyframe = """
  <keyframe>
    <key name="initial pose"
         time="0"
         qpos="{0} {1}" 
    />
  </keyframe>
""".format(panda_qpos, gripper_qpos)

task_keyframe = """
  <keyframe>
    <key name="initial pose"
         time="0"
         qpos="{0} {1}"
    />
  </keyframe>
""".format(gripper_qpos, object_qpos)

gripper_actuator_subelement = """
  <{0} name="{1}_actuator" joint="{1}"/>
"""

panda_actuator_subelement = """
  <{0} name="{1}_actuator" joint="{1}"/>
"""

finger_actuator_subelement = """
  <{0} name="{1}_actuator" joint="{1}"/>
"""

gripper_actuator_string = """"""
for joint in gripper_joints:
  gripper_actuator_string += gripper_actuator_subelement.format(
    gripper_control, joint
  )

panda_actuator_string = """"""
for joint in panda_joints:
  panda_actuator_string += panda_actuator_subelement.format(
    panda_control, joint
  )

finger_actuator_string = """"""
for joint in finger_joints:
  finger_actuator_string += finger_actuator_subelement.format(
    finger_control, joint
  )

gripper_actuator = """
  <actuator>
    {0}
    {1}
  </actuator>
""".format(gripper_actuator_string, finger_actuator_string)

panda_actuator = """
  <actuator>
    {0}
  </actuator>
""".format(panda_actuator_string)

panda_and_gripper_actuator = """
  <actuator>
    {0}
    {1}
    {2}
  </actuator>
""".format(panda_actuator_string, gripper_actuator_string, finger_actuator_string)

def modify_tag_text(tree, tagname, target_text):
  """
  This function loads an xml file, finds a specific tag, then overrides that
  tag with the given target text. These changes are saved back under the
  given filename, with the original file being now lost
  NB lxml preserves comments and ordering
  """

  # now get the root of the tree
  root = tree.getroot()

  # search recursively for all instances of the tag
  tags = root.findall(".//" + tagname)

  # now overwrite the text in each tag
  for t in tags:
    t.text = target_text

def add_tag_attribute(tree, tagname, tag_label, attribute_name, attribute_value):
  """
  Add a new attribute for a tag, eg <tag/> goes to <tag attribute="true"/>
  """

  # now get the root of the tree
  root = tree.getroot()

  # search recursively for all instances of the tag
  tags = root.findall(".//" + tagname)

  # add the attribute only if the tag_label matches
  for t in tags:
    if t.attrib["name"] == tag_label:
      t.set(attribute_name, attribute_value)

def add_chunk(tree, parent_tag, xml_string_to_add):
  """
  This function adds a chunk of xml text under the bracket of the given
  parent tag into the given tree
  """

  # extract the xml text from
  new_tree = etree.fromstring(xml_string_to_add)

  # get the root of the parent tree
  root = tree.getroot()

  # special case where we are adding at the root
  if parent_tag == "@root":
    root.append(new_tree)
    return

  # search recursively for all instances of the parent tag
  tags = root.findall(".//" + parent_tag)

  if len(tags) > 1:
    raise RuntimeError("more than one parent tag found")
  elif len(tags) == 0:
    raise RuntimeError("parent tag not found")

  for t in tags:
    t.append(new_tree)
  
  return

def add_geom_name(tree, parent_body):
  """
  Adds a name to finger segment collision geoms
  """

  # now get the root of the tree
  root = tree.getroot()

  # search recursively for all instances of the tag
  tags = root.findall(".//" + "body")

  # first geom always visual, then collision, if 4 geoms then must be hook link
  labels = ["visual", "collision", "hook_visual", "hook_collision"]

  # add the geom label only to bodies that match the parent body name
  for t in tags:
    if t.attrib["name"] == parent_body:
      geoms = t.findall("geom")
      for i, g in enumerate(geoms):
        g.set("name", parent_body + "_geom_" + labels[i])


if __name__ == "__main__":
 
  """
  This script opens a given xml file, saves the tree, and then makes some
  changes to it. The new tree then overwrites the old tree and the file is
  saved. The lxml module is used, which preserves comments and ordering, so
  the new file should look identical to the old file except the changes
  """

  directory_path = description_path + "/urdf/mujoco/"

  # define the names of the xml files we will be editing
  gripper_filename = directory_path + "gripper_mujoco.xml"
  panda_filename = directory_path + "panda_mujoco.xml"
  both_filename = directory_path + "panda_and_gripper_mujoco.xml"
  task_filename = directory_path + "gripper_task.xml"

  # parse and extract the xml tree for each
  gripper_tree = etree.parse(gripper_filename)
  panda_tree = etree.parse(panda_filename)
  both_tree = etree.parse(both_filename)
  task_tree = etree.parse(task_filename)

  # add the keyframe information to each
  add_chunk(gripper_tree, "@root", gripper_keyframe)
  add_chunk(panda_tree, "@root", panda_keyframe)
  add_chunk(both_tree, "@root", panda_and_gripper_keyframe)
  add_chunk(task_tree, "@root", task_keyframe)

  # add the actuator information to each
  add_chunk(gripper_tree, "@root", gripper_actuator)
  add_chunk(panda_tree, "@root", panda_actuator)
  add_chunk(both_tree, "@root", panda_and_gripper_actuator)
  add_chunk(task_tree, "@root", gripper_actuator)

  # now add in finger joint stiffnesses
  tag_string = "finger_{0}_segment_joint_{1}"
  body_string = "finger_{0}_segment_link_{1}"
  for i in range(3):
    for j in range(num_segments):
      next_joint = tag_string.format(i+1, j+1)
      next_body = body_string.format(i+1, j+2)
      # add finger stiffness attributes
      add_tag_attribute(gripper_tree, "joint", next_joint,
                        "stiffness", str(finger_joint_stiffness))
      add_tag_attribute(both_tree, "joint", next_joint,
                        "stiffness", str(finger_joint_stiffness))
      add_tag_attribute(task_tree, "joint", next_joint,
                        "stiffness", str(finger_joint_stiffness))
      # add geom names
      add_geom_name(task_tree, next_body)
                  
  # add the task includes
  task_includes = """<include file="objects.xml"/>"""
  add_chunk(task_tree, "worldbody", task_includes)

  # finally, overwrite the files with the new xml
  gripper_tree.write(gripper_filename, xml_declaration=True, encoding='utf-8')
  panda_tree.write(panda_filename, xml_declaration=True, encoding='utf-8')
  both_tree.write(both_filename, xml_declaration=True, encoding='utf-8')
  task_tree.write(task_filename, xml_declaration=True, encoding='utf-8')
