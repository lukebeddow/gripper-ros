#!/usr/bin/env python3

import rospy
import mujoco_py as mj
import rospkg

rospack = rospkg.RosPack()
description_path = rospack.get_path("gripper_description")
urdf_path = description_path + "//urdf//panda_and_gripper.urdf"

model = mj.load_model_from_path(urdf_path)
sim = mj.MjSim(model)
viewer = mj.MjViewer(sim)

t = 0

while True:
  sim.step()
  viewer.render()