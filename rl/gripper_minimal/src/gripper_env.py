#!/usr/bin/env python

import numpy as np
from gym import spaces
import rospy
import time
from openai_ros import robot_gazebo_env
from openai_ros.openai_ros_common import ROSLauncher

from std_msgs.msg import Float64
from std_msgs.msg import String
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose

from gazebo_msgs.srv import (SetModelConfiguration, 
                             SetModelConfigurationRequest,
                             SetModelConfigurationResponse)                         
from gripper_msgs.srv import (move_gripper, move_gripperRequest, move_gripperResponse,
                              set_pose, set_poseRequest, set_poseResponse,
                              set_joints, set_jointsRequest, set_jointsResponse)
from gripper_msgs.msg import pose as gripper_pose

from gazebo_msgs.srv import SpawnModel

class GripperEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all GripperEnv environments.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, ros_ws_abspath):
        """
        Initializes a new GripperEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered useful for AI learning.

        Sensor Topic List:
        * /wamv/odom: Odometry of the Base of Wamv

        Actuators Topic List:
        * /cmd_drive: You publish the speed of the left and right propellers.

        Args:
        """

        rospy.loginfo("Start GripperEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        self.delete_model_proxy = rospy.ServiceProxy('/gazebo/delete_model', Empty)
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

        # We launch the ROSlaunch that spawns the robot into the world
        ROSLauncher(rospackage_name = "gripper_minimal",
                    launch_file_name = "main.launch",
                    ros_ws_abspath = ros_ws_abspath)

        rospy.sleep(rospy.Duration(10))

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(GripperEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")

        rospy.loginfo("Finished setting up super class")



        rospy.loginfo("GripperEnv unpause1...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()

        # self._check_all_systems_ready()

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        # define action space
        # motor_max_displacement_mm = 5
        # base_max_displacement_mm = 5
        # max = np.array([motor_max_displacement_mm,
        #                 motor_max_displacement_mm,
        #                 motor_max_displacement_mm,
        #                 base_max_displacement_mm,
        #                 base_max_displacement_mm,
        #                 base_max_displacement_mm],
        #                 dtype = float)
        # min = np.multiply(max, -1)
        # self.action_space = spaces.Box(low = min, high = max, dtype = np.float)


        self.action_space = spaces.Discrete(12)

        # define observation space
        int_max_24bit = 8388608
        self.n_samples = 10
        max = np.array([[int_max_24bit for i in range(self.n_samples)],
                        [int_max_24bit for j in range(self.n_samples)],
                        [int_max_24bit for k in range(self.n_samples)]],
                        dtype = int)
        min = np.multiply(max, -1)
        self.observation_space = spaces.Box(low = min, high = max, dtype = np.int)

        rospy.loginfo("Defined action and observation spaces")


        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/gripper/virtual/gauge1", Float64, self._gauge1_callback)
        rospy.Subscriber("/gripper/virtual/gauge2", Float64, self._gauge2_callback)
        rospy.Subscriber("/gripper/virtual/gauge3", Float64, self._gauge3_callback)
        self.gauge1_data = []
        self.gauge2_data = []
        self.gauge3_data = []

        rospy.Subscriber("/rl/object/pose", Pose, self._object_pose_callback)

        # FOR TESTING
        rospy.Subscriber("/rl/test", String, self._testing_callback)

        # rospy.wait_for_service("/move_gripper")
        self._move_gripper_srv = rospy.ServiceProxy("/control/move_gripper", move_gripper)
        self._set_pose_srv = rospy.ServiceProxy("/control/set_pose", set_pose)
        self._set_joints_srv = rospy.ServiceProxy("/control/set_joints", set_joints)
        self._set_model_service = rospy.ServiceProxy("/gazebo/set_model_configuration", 
                                                     SetModelConfiguration)

        self._max_episode_steps = -1
        self._current_episode_steps = 0
        self.publishers_array = []
        self.object_pose = Pose()
        self.old_object_pose = Pose()

        # define the intial pose of the robot
        self.initial_pose = Pose()
        self.initial_pose.position.x = 0.5
        self.initial_pose.position.y = 0
        self.initial_pose.position.z = 0
        self.initial_pose.orientation.x = 0
        self.initial_pose.orientation.y = 1
        self.initial_pose.orientation.z = 0
        self.initial_pose.orientation.w = 0
        
        self.initial_gripper = gripper_pose()
        self.initial_gripper.x = 100e-3
        self.initial_gripper.y = 100e-3
        self.initial_gripper.z = 0.0

        # move to just above initial pose to begin
        rospy.loginfo("Instructing robot to move to above initial pose")
        initialisation_pose = Pose()
        initialisation_pose.position.x = self.initial_pose.position.x
        initialisation_pose.position.y = self.initial_pose.position.y
        initialisation_pose.position.z = self.initial_pose.position.z + 0.2
        initialisation_pose.orientation.x = self.initial_pose.orientation.x
        initialisation_pose.orientation.y = self.initial_pose.orientation.y
        initialisation_pose.orientation.z = self.initial_pose.orientation.z
        initialisation_pose.orientation.w = self.initial_pose.orientation.w

        request = set_poseRequest()
        response = set_poseResponse()

        request.panda_pose = initialisation_pose
        request.gripper_pose = self.initial_gripper
        request.ignore_panda = False
        request.ignore_gripper = False

        try:
            response = self._set_pose_srv(request)
        except rospy.ServiceException as exc:
            print("Service did not proces request: " + str(exc))

        rospy.loginfo("Robot now at initial pose")

        # self._cmd_drive_pub = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)

        # self.publishers_array.append(self._cmd_drive_pub)

        # self._check_all_publishers_ready()

        self.gazebo.pauseSim()

        rospy.loginfo("Finished GripperEnv INIT...")

    # ----- Virtual functions from base class (RobotGazeboEnv) to be overloaded ----- #
    def _reset_sim(self):
        """Resets a simulation
        """
        rospy.logdebug("RESET SIM START")
        if self.reset_controls :
            rospy.logdebug("RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()                  # TODO
            self._set_init_pose()                            # TODO
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self.controllers_object.reset_controllers()
            self._check_all_systems_ready()                  # TODO
            self.gazebo.pauseSim()

        else:
            rospy.logwarn("DONT RESET CONTROLLERS")
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self._set_init_pose()
            self.gazebo.pauseSim()
            self.gazebo.resetSim()
            self.gazebo.unpauseSim()
            self._check_all_systems_ready()
            self.gazebo.pauseSim()

        rospy.logdebug("RESET SIM END")
        return True

    def _set_init_pose(self):

        rospy.loginfo("Instructing robot to move to initial pose")

        request = set_poseRequest()
        response = set_poseResponse()

        request.panda_pose = self.initial_pose
        request.gripper_pose = self.initial_gripper
        request.ignore_panda = False
        request.ignore_gripper = False

        try:
            response = self._set_pose_srv(request)
        except rospy.ServiceException as exc:
            print("Service did not proces request: " + str(exc))

        rospy.loginfo("Robot now at initial pose")

    def _set_init_pose_FAILED(self):
        """Sets the Robot in its init pose
        """

        msg = SetModelConfigurationRequest()
        resp = SetModelConfigurationResponse()

        joints_to_set = ["panda_joint0",
                         "panda_joint1",
                         "panda_joint2",
                         "panda_joint3",
                         "panda_joint4",
                         "panda_joint5",
                         "panda_joint6",
                         "panda_joint7",
                         "finger_1_prismatic_joint",
                         "finger_1_revolute_joint",
                         "finger_2_prismatic_joint",
                         "finger_2_revolute_joint",
                         "finger_3_prismatic_joint",
                         "finger_3_revolute_joint",
                         "palm_prismatic_joint"]

        joint_values = [0.2,    # panda_joint0
                        0.3,    # panda_joint1
                        0.4,    # panda_joint2
                        0.0,    # panda_joint3
                        0.0,    # panda_joint4
                        0.3,    # panda_joint5
                        1.0,    # panda_joint6
                        0.0,    # panda_joint7
                        130e-3, # finger_1_prismatic_joint
                        0.0,    # finger_1_revolute_joint
                        130e-3, # finger_2_prismatic_joint
                        0.0,    # finger_2_revolute_joint
                        130e-3, # finger_3_prismatic_joint
                        0.0,    # finger_3_revolute_joint
                        0.0]    # palm_prismatic_joint

        msg.model_name = "panda_and_gripper"
        msg.urdf_param_name = "robot_description"
        msg.joint_names = joints_to_set
        msg.joint_positions = joint_values

        try:
            resp = self._set_model_service(msg)
        except rospy.ServiceException as exc:
            print("Service did not proces request: " + str(exc))

        rospy.logwarn("Model pose reset gave bool {}".format(resp.success))
        
        if not resp.success:
            rospy.logerr("Model pose failed, msg: {}".format(resp.status_message))

        return

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        timeout = 10.0
        ready = True
        # check gauges
        ready *= self._check_gauge_topics(timeout)
        ready *= self._check_object_topics(timeout)
        # check action service
        try:
            self._move_gripper_srv.wait_for_service(timeout)
        except rospy.ServiceException as e:
            print("check of service failed: {}".format(e))
            ready *= False
        # add more checks later
        return ready

    def _get_obs(self):
        """Returns the observation.
        """

        # return gauge data for each gauge from the last no. of samples
        obs = np.array([[self.gauge1_data[-self.n_samples:]],
                        [self.gauge2_data[-self.n_samples:]],
                        [self.gauge3_data[-self.n_samples:]]],
                        dtype = int)

        return obs

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        self.gauge1_data = []
        self.gauge2_data = []
        self.gauge3_data = []
        self._current_episode_steps = 0

        return

    def _set_action(self,action):

        arm_move_mm = 5

        x_nudge = 5e-3
        y_nudge = 5e-3
        z_nudge = 5e-3

        msg = move_gripperRequest()
        resp = move_gripperResponse()

        # # for testing, limit possible actions to 1, 3, 5
        action %= 6
        rospy.logwarn("Action is {}".format(action))

        if action == 0:
            msg.nudge.gripper.x = x_nudge
        elif action == 1:
            msg.nudge.gripper.x = -x_nudge
        elif action == 2:
            msg.nudge.gripper.y = y_nudge
        elif action == 3:
            msg.nudge.gripper.y = -y_nudge
        elif action == 4:
            msg.nudge.gripper.z = z_nudge
        elif action == 5:
            msg.nudge.gripper.z = -z_nudge
        elif action == 6:
            msg.nudge.arm.x = arm_move_mm * 1e-3
        elif action == 7:
            msg.nudge.arm.x = -arm_move_mm * 1e-3
        elif action == 8:
            msg.nudge.arm.y = arm_move_mm * 1e-3
        elif action == 9:
            msg.nudge.arm.y = -arm_move_mm * 1e-3
        elif action == 10:
            msg.nudge.arm.z = arm_move_mm * 1e-3
        elif action == 11:
            msg.nudge.arm.z = -arm_move_mm * 1e-3

        # take this opportunity to update the saved object pose
        self.old_object_pose = self.object_pose

        self._current_episode_steps += 1

        try:
            resp = self._move_gripper_srv(msg)
        except rospy.ServiceException as exc:
            print("Service did not proces request: " + str(exc))

        rospy.sleep(rospy.Duration(5))
        
        return

    def _set_action_BOX(self, action):
        """Applies the given action to the simulation.
        """

        msg = move_gripperRequest()
        resp = move_gripperResponse()

        self.old_object_pose = self.object_pose

        msg.nudge.gripper.x = action[0]
        msg.nudge.gripper.th = action[1]
        msg.nudge.gripper.z = action[2]
        msg.nudge.arm.x = action[3]
        msg.nudge.arm.y = action[4]
        msg.nudge.arm.z = action[5]
        # msg.nudge.arm.roll = 0.1
        # msg.nudge.arm.pitch = 0
        # msg.nudge.arm.yaw = 0

        try:
            resp = self._move_gripper_srv(msg)
        except rospy.ServiceException as exc:
            print("Service did not proces request: " + str(exc))
        
        return

    def set_max_steps(self, max):
        self._max_episode_steps = max

    def _is_done(self, observations):
        """Indicates whether or not the episode is done (the robot has fallen for example).
        """

        # self._max_episode_steps = 5
        rospy.logwarn("current_episode_steps: {0}, max_episode_steps: {1}".format(
            self._current_episode_steps, self._max_episode_steps))

        if self._current_episode_steps == self._max_episode_steps:
            return True

        return False

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        # reward based only on z lift, 1point per mm
        reward = 1000 * (self.object_pose.position.z - self.old_object_pose.position.z)
        return reward

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        # do nothing for now
        return

    # ----- Callback functions for reading topic data ----- #

    # gauge data callbacks
    def _gauge1_callback(self, msg):
        self.gauge1_data.append(msg.data)
    def _gauge2_callback(self, msg):
        self.gauge2_data.append(msg.data)
    def _gauge3_callback(self, msg):
        self.gauge3_data.append(msg.data)

    # object data callbacks
    def _object_pose_callback(self, msg):
        self.object_pose = msg

    # FOR TESTING
    def _testing_callback(self, msg):
        if msg.data == "obs":
            obs = self._get_obs()
            print("obs is:\n", obs)
        elif msg.data == "reward":
            obs = self._get_obs
            done = False
            reward = self._compute_reward(obs, done)
            print("The reward is:", reward)
        elif msg.data == "action1":
            print("Setting action")
            action = [10, 0, 0, 0, 0, 0]
            self._set_action(action)
        elif msg.data == "action2":
            print("Setting action")
            action = [-10, 0, 0, 0, 0, 0]
            self._set_action(action)
        elif msg.data == "action3":
            print("Setting action")
            action = [0, -10, 0, 0, 0, 0]
            self._set_action(action)
        elif msg.data == "action4":
            print("Setting action")
            action = [0, 10, 0, 0, 0, 0]
            self._set_action(action)
        elif msg.data == "action5":
            print("Setting action")
            action = [0, 0, 0, 0.1, 0, 0]
            self._set_action(action)
        elif msg.data == "action6":
            print("Setting action")
            action = [0, 0, 0, -0.1, 0, 0]
            self._set_action(action)

    # ----- Helper functions ----- #
    def _check_gauge_topics(self, timeout=1.0):
        try:
            rospy.wait_for_message("/gripper/virtual/gauge1", Float64, timeout)
            rospy.wait_for_message("/gripper/virtual/gauge2", Float64, timeout)
            rospy.wait_for_message("/gripper/virtual/gauge3", Float64, timeout)
            return True
        except rospy.exceptions.ROSException:
            pass
        return False

    def _check_object_topics(self, timeout=1.0):
        try:
            rospy.wait_for_message("/rl/object/pose", Pose, timeout)
            return True
        except rospy.exceptions.ROSException:
            pass
        return False

if __name__ == "__main__":

    rospy.init_node("gripper_env_node")

    ros_ws_abspath = "/home/luke/gripper_repo_ws"

    test = GripperEnv(ros_ws_abspath)

    rospy.spin()