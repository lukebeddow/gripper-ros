#!/usr/bin/env python

import pybullet as p
import rospy
import rospkg

"""
link name: panda_link0, id: 0
link name: panda_link1, id: 1
link name: panda_link2, id: 2
link name: panda_link3, id: 3
link name: panda_link4, id: 4
link name: panda_link5, id: 5
link name: panda_link6, id: 6
link name: panda_link7, id: 7
link name: panda_link8, id: 8
link name: gripper_base_link, id: 9
link name: finger_1_intermediate, id: 10
link name: finger_1, id: 11
link name: finger_1_segment_link_1, id: 12
link name: finger_1_segment_link_2, id: 13
link name: finger_1_segment_link_3, id: 14
link name: finger_1_segment_link_4, id: 15
link name: finger_1_segment_link_5, id: 16
link name: finger_1_segment_link_6, id: 17
link name: finger_1_segment_link_7, id: 18
link name: finger_1_segment_link_8, id: 19
link name: finger_1_segment_link_9, id: 20
link name: finger_1_segment_link_10, id: 21
link name: finger_1_finger_hook_link, id: 22
link name: finger_2_intermediate, id: 23
link name: finger_2, id: 24
link name: finger_2_segment_link_1, id: 25
link name: finger_2_segment_link_2, id: 26
link name: finger_2_segment_link_3, id: 27
link name: finger_2_segment_link_4, id: 28
link name: finger_2_segment_link_5, id: 29
link name: finger_2_segment_link_6, id: 30
link name: finger_2_segment_link_7, id: 31
link name: finger_2_segment_link_8, id: 32
link name: finger_2_segment_link_9, id: 33
link name: finger_2_segment_link_10, id: 34
link name: finger_2_finger_hook_link, id: 35
link name: finger_3_intermediate, id: 36
link name: finger_3, id: 37
link name: finger_3_segment_link_1, id: 38
link name: finger_3_segment_link_2, id: 39
link name: finger_3_segment_link_3, id: 40
link name: finger_3_segment_link_4, id: 41
link name: finger_3_segment_link_5, id: 42
link name: finger_3_segment_link_6, id: 43
link name: finger_3_segment_link_7, id: 44
link name: finger_3_segment_link_8, id: 45
link name: finger_3_segment_link_9, id: 46
link name: finger_3_segment_link_10, id: 47
link name: finger_3_finger_hook_link, id: 48
link name: palm, id: 49
"""

class RobotEnv():

  def __init__(self):

    # get the path to the urdf
    rospack = rospkg.RosPack()
    description_path = rospack.get_path("gripper_description")
    self.urdf_path = description_path + "//urdf//panda_and_gripper.urdf"

  def load(self):
    """
    Load the robot into pybullet
    """

    # define starting pose of the robot
    start_xyz = [0, 0, 0]
    start_rpy = [0, 0, 0]
    start_quaternion = p.getQuaternionFromEuler(start_rpy)

    # load into the simulation
    self.robot_id = p.loadURDF(self.urdf_path, start_xyz, start_quaternion)

    # create data structure for setting the joints to specific values
    self.joints = {
      # panda arm
      "panda_joint1" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint2" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint3" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint4" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint5" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint6" : { "value" : 0.0, "id" : -1, "set" : False },
      "panda_joint7" : { "value" : 0.0, "id" : -1, "set" : False },
      # gripper
      "finger_1_prismatic_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_prismatic_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_prismatic_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_revolute_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_revolute_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_revolute_joint" : { "value" : 0.0, "id" : -1, "set" : False },
      "palm_prismatic_joint" : { "value" : 0.0, "id" : -1, "set" : False },
    }

    # create a seperate finger joints structure
    self.finger_joints = {
      # finger 1
      "finger_1_segment_joint_1" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_2" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_3" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_4" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_5" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_6" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_7" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_8" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_1_segment_joint_9" : { "value" : 0.0, "id" : -1, "set" : False },
      # finger 2
      "finger_2_segment_joint_1" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_2" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_3" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_4" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_5" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_6" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_7" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_8" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_2_segment_joint_9" : { "value" : 0.0, "id" : -1, "set" : False },
      # finger 3
      "finger_3_segment_joint_1" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_2" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_3" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_4" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_5" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_6" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_7" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_8" : { "value" : 0.0, "id" : -1, "set" : False },
      "finger_3_segment_joint_9" : { "value" : 0.0, "id" : -1, "set" : False },
    }

    # input data about the joint ids
    self.save_joint_info()

    # disable the default pybullet velocity motors
    self.set_finger_motors(max_force=0.0, control="velocity", target=0)

  def get_link_info(self):
    """
    Extract information about the joints
    """

    # create a dictionary to move from link names to indices
    self.link_name_to_index = {p.getBodyInfo(self.robot_id)[0].decode('UTF-8') : -1,}

    # fill the dictionary with every link name
    for id in range(p.getNumJoints(self.robot_id)):
      name = p.getJointInfo(self.robot_id, id)[12].decode('UTF-8')
      self.link_name_to_index[name] = id

  def get_joint_id(self, name):
    """
    Get the joint id using the name of the joint
    """

    # for the panda arm joints
    if name[:5] == "panda":
      return self.link_name_to_index["panda_link" + name[-1]]

    # for the gripper joints
    if name[9:] == "prismatic_joint":
      return self.link_name_to_index[name[:8] + "_intermediate"]
    if name[9:] == "revolute_joint":
      return self.link_name_to_index[name[:8]]
    if name[:4] == "palm":
      return self.link_name_to_index[name[:4]]

    # for the segmented fingers
    if name[9:22] == "segment_joint":
      j_name = "finger_" + name[7] + "_segment_link_" + str(int(name[-1]) + 1)
      # print(p.getJointInfo(self.robot_id, self.link_name_to_index[j_name]))
      return self.link_name_to_index[j_name]

  def save_joint_info(self):
    """
    This function saves information about the joints
    """

    self.finger_id_list = []

    # first save link info from pybullet
    self.get_link_info()

    # now store the id of each joint
    for joint in self.joints:
      id = self.get_joint_id(joint)
      self.joints[joint]["id"] = id

    # do the same for the finger joints
    for f_joint in self.finger_joints:
      id = self.get_joint_id(f_joint)
      self.finger_joints[f_joint]["id"] = id
      self.finger_id_list.append(id)

  def set_finger_motors(self, max_force=0.0, control="position", target=0):
    """
    This function sets the finger motors, setting max_force=0 disables them
    """

    if control == "torque":
      mode = p.TORQUE_CONTROL
      for joint in self.finger_joints:
        p.setJointMotorControl2(bodyUniqueId = self.robot_id,
                                jointIndex = self.finger_joints[joint]["id"],
                                controlMode = mode,
                                force = max_force)

    elif control == "position":
      mode = p.POSITION_CONTROL
      for joint in self.finger_joints:
        p.setJointMotorControl2(bodyUniqueId = self.robot_id,
                                jointIndex = self.finger_joints[joint]["id"],
                                controlMode = mode,
                                force = max_force,
                                targetPosition = target)

    elif control == "velocity":
      mode = p.VELOCITY_CONTROL
      for joint in self.finger_joints:
        p.setJointMotorControl2(bodyUniqueId = self.robot_id,
                                jointIndex = self.finger_joints[joint]["id"],
                                controlMode = mode,
                                force = max_force,
                                targetVelocity = target)

    else:
      raise RuntimeError("motor control type can be torque, position, or velocity only")

  def set_joint_positions(self):
    """
    Sets the joints to the values given in self.joints, if set = True
    """

    # loop through the joints and set them to the given positions
    for joint in self.joints:
      if self.joints[joint]["set"]:
        p.resetJointState(self.robot_id,
                          self.joints[joint]["id"],
                          self.joints[joint]["value"])
        # now we have set it, we don't want to set twice
        self.joints[joint]["set"] = False

  def set_initial_pose(self):
    """
    Set the robot into its inital pose
    """

    # hardcode the initial pose
    start_pose = {
      "panda_joint1" : 0.0,
      "panda_joint2" : 0.0,
      "panda_joint3" : 0.0,
      "panda_joint4" : 0.0,
      "panda_joint5" : 0.0,
      "panda_joint6" : 1.0,
      "panda_joint7" : 0.0,
      "finger_1_prismatic_joint" : 140e-3,
      "finger_2_prismatic_joint" : 140e-3,
      "finger_3_prismatic_joint" : 140e-3,
      "finger_1_revolute_joint" : 0.0,
      "finger_2_revolute_joint" : 0.0,
      "finger_3_revolute_joint" : 0.0,
      "palm_prismatic_joint" : 0e-3
    }

    for joint_name, value in start_pose.items():
      self.joints[joint_name]["value"] = value
      self.joints[joint_name]["set"] = True

    # set the joint positions in the simulator
    self.set_joint_positions()

  def calculate_finger_torques(self):
    """
    This function calculates the torque for each segmented finger joint
    """

    # setup variables
    finger_string = "finger_{0}_segment_link_{1}"
    finger_positions = [[0 for x in range(9)] for y in range(3)]
    finger_velocities = [[0 for x in range(9)] for y in range(3)]
    finger_torques = [[0 for x in range(9)] for y in range(3)]

    stiffness = 0.5     # Nm per rad
    damping = 0.0001    # N per m/s
    friction = 0        # N/m

    # loop through each finger, and each joint
    for i in range(3):
      for j in range(9):
        id = self.link_name_to_index[finger_string.format(i + 1, j + 2)]
        pos, vel, forces, motor = p.getJointState(self.robot_id, id)

        finger_positions[i][j] = pos

        if j == 0:
          kx = pos * stiffness
          cxdot = vel * damping
        else:
          kx = (pos) * stiffness
          cxdot = (vel) * damping

        torque = (-kx) + (-cxdot)

        if abs(torque) < friction:
          torque = 0

        old_pos = pos
        old_vel = vel

        p.applyExternalTorque(objectUniqueId = self.robot_id, 
                              linkIndex = id, 
                              torqueObj = [0.0, torque, 0.0], 
                              flags = p.LINK_FRAME)
                          
    # print("finger positions:", finger_positions)

