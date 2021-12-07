#!/usr/bin/env python

from __future__ import print_function

import pybullet as p
import time
import pybullet_data

import robot_env

# start pybullet and load in the robot
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

timestep = 1./2400. # default 240Hz (1./240.)
p.setTimeStep(timestep)


# create robot object
robot = robot_env.RobotEnv()
robot.load()

robot.set_initial_pose()

# dummy = raw_input("press enter")

for i in range(30000):
  p.stepSimulation()
  # robot.set_finger_motors(max_force=0.01, control="torque")
  robot.calculate_finger_torques()
  time.sleep(timestep)
  print("iteration {}".format(i), end="\r")
  # dummy = raw_input("press enter")

print("Finished all iterations")
cubePos, cubeOrn = p.getBasePositionAndOrientation(robot.robot_id)
print(cubePos, cubeOrn)
p.disconnect()