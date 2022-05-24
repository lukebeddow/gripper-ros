#!/usr/bin/env python

import yaml
import rospkg
import lxml.etree as etree

# open the grippr details yaml file to extract parameters
rospack = rospkg.RosPack()
description_path = rospack.get_path("gripper_description")
with open(description_path + "//config//gripper.yaml") as file:
  gripper_details = yaml.safe_load(file)

# exctract the details of the gripper configuration
is_segmented = gripper_details["gripper_config"]["is_segmented"]
num_segments = gripper_details["gripper_config"]["num_segments"]

# panda parameters
panda_control = "motor"

# gripper parameters
gripper_control = "position"
force_limit_prismatic = 1.0
force_limit_revolute = 1.0
force_limit_palm = 1.0

# starting configuration of the robot joints
j = {
  "panda_joint1": 0,
  "panda_joint2": 0,
  "panda_joint3": 0,
  "panda_joint4": 0,
  "panda_joint5": 0,
  "panda_joint6": 1.0,
  "panda_joint7": 0,
  "gripper_prismatic": 140e-3,
  "gripper_revolute": 0,
  "gripper_palm": 0
}

# now automatically generate xml strings to encode these starting values
if is_segmented:
  finger_joints = "0 " * (num_segments - 1)
else:
  finger_joints = ""

gripper_qpos = "{0} {1} {2} {0} {1} {2} {0} {1} {2} {3}".format(
  j["gripper_prismatic"], j["gripper_revolute"], finger_joints, j["gripper_palm"]
)
panda_qpos = "{0} {1} {2} {3} {4} {5} {6}".format(
  j["panda_joint1"], j["panda_joint2"], j["panda_joint3"], j["panda_joint4"],
  j["panda_joint5"], j["panda_joint6"], j["panda_joint7"]
)

all_compiler = """
  <compiler
    meshir="./meshes_mujoco/"
    balanceinertia="true"
    discardvisual="false"
  />
"""

gripper_size = """
  <size njmax="500" nconmax="100"/>
"""

panda_size = """
  <size njmax="500" nconmax="100"/>
"""

panda_and_gripper_size = """
  <size njmax="1400" nconmax="500"/>
"""

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

  tags.append(new_tree)

  # # now add the chunk to every parent
  # import copy
  # for t in tags:
  #   tags.append(copy.deepcopy(new_tree))


if __name__ == "__main__":
 
  """
  This script opens a given xml file, saves the tree, and then makes some
  changes to it. The new tree then overwrites the old tree and the file is
  saved. The lxml module is used, which preserves comments and ordering, so
  the new file should look identical to the old file except the changes
  """

  directory_path = description_path + "//urdf//mujoco//"

  # define the names of the xml files we will be editing
  gripper_filename = directory_path + "gripper_mujoco.xml"
  panda_filename = directory_path + "panda_mujoco.xml"
  both_filename = directory_path + "panda_and_gripper_mujoco.xml"

  # parse and extract the xml tree for each
  gripper_tree = etree.parse(gripper_filename)
  panda_tree = etree.parse(panda_filename)
  both_tree = etree.parse(both_filename)

  # add the keyframe information to each
  add_chunk(gripper_tree, "@root", gripper_keyframe)
  add_chunk(panda_tree, "@root", panda_keyframe)
  add_chunk(both_tree, "@root", panda_and_gripper_keyframe)

  # now add in finger joint stiffnesses
  tag_string = "finger_{0}_segment_joint_{1}"
  stiffness = 1.0
  for i in range(3):
    for j in range(num_segments):
      add_tag_attribute(gripper_tree, "joint", tag_string.format(i+1, j+1),
                        "stiffness", str(stiffness))
      add_tag_attribute(both_tree, "joint", tag_string.format(i+1, j+1),
                        "stiffness", str(stiffness))

  # finally, overwrite the files with the new xml
  gripper_tree.write(gripper_filename, xml_declaration=True, encoding='utf-8')
  panda_tree.write(panda_filename, xml_declaration=True, encoding='utf-8')
  both_tree.write(both_filename, xml_declaration=True, encoding='utf-8')
