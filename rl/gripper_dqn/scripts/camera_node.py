#!/home/luke/pyenv/py38_ros/bin/python

import rospy
from gripper_msgs.msg import RGBD
from sensor_msgs.msg import Image

# try:
#   depth_camera_connected = True
#   from capture_depth_image import get_depth_image
# except Exception as e: 
#   rospy.logerr("DEPTH CAMERA NOT CONNECTED")
#   rospy.logerr(f"Depth camera error message: {e}")
#   depth_camera_connected = False

run_this_node = False
if not run_this_node:
  print("CAMERA_NODE.PY QUITTING BEFORE STARTING, run_this_node = False")
  exit()

depth_camera_connected = True

import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

device_id = None #"128422272085"  # "923322071108" # serial number of device to use or None to use default

framerate = 30
width = 640 #848
height = 360

pipeline = rs.pipeline()
config = rs.config()

# if we are provided with a specific device, then enable it
if None != device_id:
    config.enable_device(device_id)

config.enable_stream(rs.stream.depth, width, height, rs.format.z16, framerate)  # depth
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, framerate)  # rgb

# Start streaming
profile = pipeline.start(config)

# causes error: first frame never comes even after 10 seconds
# depth_sensor = profile.get_device().query_sensors()[0]
# depth_sensor.set_option(rs.option.enable_auto_exposure, False)

# index error: list out of range
# rgb_sensor = profile.get_device().query_sensors()[1]
# rgb_sensor.set_option(rs.option.enable_auto_exposure, False)

# test: https://github.com/IntelRealSense/librealsense/issues/5885
# profile.get_device().query_sensors()[1].set_option(rs.option.auto_exposure_priority, 0.0)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: ", depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

frame_count = 0
start_time = time.time()
frame_time = start_time

def get_depth_image(arg1=None, arg2=None, arg3=None):
  """
  code from: https://github.com/IntelRealSense/librealsense/issues/5628#issuecomment-575943238
  """

  global frame_count

  wait_ms = 1000

  if frame_count == 0:
    frames = pipeline.wait_for_frames(10000) # wait 10 seconds for first frame
    rospy.loginfo("camera_node has received the first frame successfully")
  else: 
    frames = pipeline.wait_for_frames(wait_ms)

  # Align the depth frame to color frame
  aligned_frames = align.process(frames)
  depth_frame = aligned_frames.get_depth_frame()
  color_frame = aligned_frames.get_color_frame()

  # Convert images to numpy arrays
  depth_image = np.asanyarray(depth_frame.get_data())
  color_image = np.asanyarray(color_frame.get_data())

  frame_count += 1

  if frame_count > 10_000_000: frame_count = 10

  return color_image, depth_image

if __name__ == "__main__":

  if not depth_camera_connected:
    rospy.logwarn("Camera node failed to start, no depth camera connected")
    exit()

  # initilise ros
  rospy.init_node("camera_node")
  rospy.loginfo("camera node has now started")

  bridge = CvBridge()
  
  # create the publisher
  node_ns = "camera" # gripper/camera
  rgb_pub = rospy.Publisher(f"/{node_ns}/rgb", Image, queue_size=10)
  depth_pub = rospy.Publisher(f"/{node_ns}/depth", Image, queue_size=10)

  r = rospy.Rate(20) # 20hz 

  while not rospy.is_shutdown():

    rgb, depth = get_depth_image()
    img_msg = bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
    depth_msg = bridge.cv2_to_imgmsg(depth, encoding="16UC1")

    rgb_pub.publish(img_msg)
    depth_pub.publish(depth_msg)

    r.sleep()