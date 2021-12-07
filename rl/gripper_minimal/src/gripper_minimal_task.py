#!/usr/bin/env python

import rospy
import numpy as np
from gym import spaces
# from openai_ros.robot_envs import wamv_env
from gripper_env import GripperEnv
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os
from gripper_msgs.srv import move_gripper, move_gripperRequest, move_gripperResponse

class GripperTask(GripperEnv):
    def __init__(self):
        """
        Teach the gripper to pick up an object
        """

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="gripper_minimal",
                               rel_path_from_package_to_file="config",
                               yaml_file_name="gripper_task.yaml")

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/gripper/ros_ws_abspath", None)




        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        # ROSLauncher(rospackage_name="robotx_gazebo",
        #             launch_file_name="start_world.launch",
        #             ros_ws_abspath=ros_ws_abspath)

        

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(GripperTask, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here

        rospy.logdebug("Start GripperTask INIT...")

        # Get parameters from yaml file
        self.example_param_1 = rospy.get_param('/gripper/example_param_1')
        self.example_param_1 = rospy.get_param('/gripper/example_param_1')

        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        # Rewards

        # self.done_reward =rospy.get_param("/wamv/done_reward")
        # self.closer_to_point_reward = rospy.get_param("/wamv/closer_to_point_reward")

        # self.cumulated_steps = 0.0

        rospy.logdebug("END GripperTask INIT...")

    # def _set_init_pose(self):
    #     """
    #     Sets the two proppelers speed to 0.0 and waits for the time_sleep
    #     to allow the action to be executed
    #     """

    #     pass

    #     return True


    # def _init_env_variables(self):
    #     """
    #     Inits variables needed to be initialised each time we reset at the start
    #     of an episode.
    #     :return:
    #     """

    #     # For Info Purposes
    #     self.cumulated_reward = 0.0
    #     # We get the initial pose to mesure the distance from the desired point.
    #     odom = self.get_odom()
    #     current_position = Vector3()
    #     current_position.x = odom.pose.pose.position.x
    #     current_position.y = odom.pose.pose.position.y
    #     self.previous_distance_from_des_point = self.get_distance_from_desired_point(current_position)


    # def _set_action(self, action):
    #     """
    #     It sets the joints of wamv based on the action integer given
    #     based on the action number given.
    #     :param action: The action integer that sets what movement to do next.
    #     """

    #     rospy.logdebug("Start Set Action ==>"+str(action))

    #     rospy.logdebug("END Set Action ==>"+str(action))

    # def _get_obs(self):
    #     """
    #     Here we define what sensor data defines our robots observations
    #     To know which Variables we have access to, we need to read the
    #     WamvEnv API DOCS.
    #     :return: observation
    #     """
    #     rospy.logdebug("Start Get Observation ==>")

    #     return observation


    # def _is_done(self, observations):
    #     """
    #     We consider the episode done if:
    #     1) The ball has been lifted and then dropped again
    #     2) The ball is outside the gripper workspace
    #     """
    #     distance_from_desired_point = observations[8]

    #     current_position = Vector3()
    #     current_position.x = observations[0]
    #     current_position.y = observations[1]

    #     is_inside_corridor = self.is_inside_workspace(current_position)
    #     has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)

    #     done = not(is_inside_corridor) or has_reached_des_point

    #     return done

    # def _compute_reward(self, observations, done):
    #     """
    #     We Base the rewards in if its done or not and we base it on
    #     if the distance to the desired point has increased or not
    #     :return:
    #     """

    #     # We only consider the plane, the fluctuation in z is due mainly to wave
    #     current_position = Point()
    #     current_position.x = observations[0]
    #     current_position.y = observations[1]

    #     distance_from_des_point = self.get_distance_from_desired_point(current_position)
    #     distance_difference =  distance_from_des_point - self.previous_distance_from_des_point


    #     if not done:

    #         # If there has been a decrease in the distance to the desired point, we reward it
    #         if distance_difference < 0.0:
    #             rospy.logwarn("DECREASE IN DISTANCE GOOD")
    #             reward = self.closer_to_point_reward
    #         else:
    #             rospy.logerr("ENCREASE IN DISTANCE BAD")
    #             reward = -1*self.closer_to_point_reward

    #     else:

    #         if self.is_in_desired_position(current_position, self.desired_point_epsilon):
    #             reward = self.done_reward
    #         else:
    #             reward = -1*self.done_reward


    #     self.previous_distance_from_des_point = distance_from_des_point


    #     rospy.logdebug("reward=" + str(reward))
    #     self.cumulated_reward += reward
    #     rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
    #     self.cumulated_steps += 1
    #     rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

    #     return reward


  
