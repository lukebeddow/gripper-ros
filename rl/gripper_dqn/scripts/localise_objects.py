#!/home/luke/pyenv/py38_ros/bin/python

# general imports
import time
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2

# ros imports
import rospy
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from gripper_dqn.msg import object_positions

# global variables
rgb_image = None
depth_image = None
camera = True
depth_camera_connected = False
xyz_points = []
object_names = []
object_confidences = []

# bin wall positions on depth image
wall_x_pixel_low = 105
wall_x_pixel_high = 545
wall_y_pixel_low = 55
wall_y_pixel_high = 320

# thresholds for removing floor and outliers from depth
low_threshold = 10
up_threshold = 15

# bin corner positions (d is not used)
a_metres = (0.204, 0.013)
b_metres = (0.211, 0.317)
c_metres = (0.742, 0.049)
d_metres = (0.749, 0.329)

# create openCV bridge
bridge = CvBridge()

if False:

  # Model
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

  # Images
  imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images

  # Inference
  results = model(imgs)

  # Results
  results.print()
  # results.save()  # or .show()
  results.show()

  results.xyxy[0]  # img1 predictions (tensor)
  results.pandas().xyxy[0]  # img1 predictions (pandas)
  #      xmin    ymin    xmax   ymax  confidence  class    name
  # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
  # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
  # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
  # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

  def localise_objects():
    """
    Return the xyz points of objects, and their names from YOLO
    """

    # wipe any pre-existing localisation information
    xyz_points = []
    object_names = []
    object_confidences = [] 

    # get the most recent rgb and depth images from ros topics
    rgb, d = get_depth_image()

    # run inference on the rgb image
    infer = model(rgb)

    infer.print()

    infer.show()

    # for testing
    xyz_points.append([0, 1, 2])
    xyz_points.append([-1, 1, 0])
    object_names.append("test1")
    object_names.append("test2")
    object_confidences.append(0.56)
    object_confidences.append(0.78)

def connected_callback(msg):
  """
  Check if the camera is connected
  """
  global depth_camera_connected
  depth_camera_connected = msg.data

def rgb_callback(msg):
  """
  Save the most recent rgb image
  """
  global rgb_image
  cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
  rgb_image = np.array(cv_image, dtype=np.uint8)

def depth_callback(msg):
  """
  Save the most recent depth image
  """

  global depth_image
  cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
  depth_image = np.array(cv_image, dtype=np.float32)

def get_depth_image():
  """
  Function returning the most recent rgb, depth image
  """
  global depth_camera_connected, camera

  if not camera: return None, None

  if rgb_image is None or depth_image is None:
    rospy.logwarn("Warning: camera images are None")
  else:
    depth_camera_connected = True

  return rgb_image, depth_image

def standard_depth_process(img):
  """
  Process depth images
  """

  img /= 256

  # apply thresholding
  img[img > up_threshold] = 0
  img[img < low_threshold] = 0

  wall_x_pixel_low = 105
  wall_x_pixel_high = 545
  wall_y_pixel_low = 55
  wall_y_pixel_high = 320

  # cut out the walls (set to zero)
  img[:, :wall_x_pixel_low] = 0
  img[:, wall_x_pixel_high:] = 0
  img[:wall_y_pixel_low, :] = 0
  img[wall_y_pixel_high:, :] = 0
  
  # make binary
  img[img > 1] = 7

  img = np.array(img, dtype=np.uint8)

  return img

def gptfcn(binary_image):

  # Step 3: Find connected components
  num_labels, labels_im = cv2.connectedComponents(binary_image)

  # Prepare to filter small components
  min_size = 500  # Adjust this value based on the expected size of your blobs
  centroids = []

  # Step 4: Filter out small components and calculate centroids
  for label in range(1, num_labels):  # Skipping the background label (0)
      component_mask = (labels_im == label)
      if np.sum(component_mask) >= min_size:
          M = cv2.moments(component_mask.astype(np.uint8))
          if M["m00"] != 0:
              cX = int(M["m10"] / M["m00"])
              cY = int(M["m01"] / M["m00"])
              centroids.append((cX, cY))

  # Step 5: Plot centroids on the original image
  output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
  output_image *= 30
  for centroid in centroids:
      cv2.circle(output_image, centroid, 5, (0, 0, 255), -1)

  return output_image, centroids

def pixels_to_metres(centroids):
  """
  Convert pixel locations in the depth image into metre positions

  panda

  a -------- b
  |          |
  |          |
  |          |
  |          |
  |          |
  c -------- d
  """

  a_pixel = (wall_x_pixel_low, wall_y_pixel_high)
  b_pixel = (wall_x_pixel_high, wall_y_pixel_high)
  c_pixel = (wall_x_pixel_low, wall_y_pixel_low)
  d_pixel = (wall_x_pixel_high, wall_y_pixel_low)

  # change in A with respect to value of B
  x_wrt_y = (b_metres[0] - a_metres[0])
  x_wrt_x = (c_metres[0] - a_metres[0])
  y_wrt_x = (c_metres[1] - a_metres[1])
  y_wrt_y = (b_metres[1] - a_metres[1])

  centroids_metres = []

  # convert with linear interpolation
  for cx, cy in centroids:

    # clamp in allowable range
    if cx < wall_x_pixel_low: cx = wall_x_pixel_low
    if cx > wall_x_pixel_high: cx = wall_x_pixel_high
    if cy < wall_y_pixel_low: cy = wall_y_pixel_low
    if cy > wall_y_pixel_high: cy = wall_y_pixel_high

    # move pixels to start at 0, 0
    cx -= wall_x_pixel_low
    cy -= wall_y_pixel_low

    # convert to factor [0.0, 1.0] of workspace
    cx /= (wall_x_pixel_high - wall_x_pixel_low)
    cy /= (wall_y_pixel_high - wall_y_pixel_low)

    # note x for pixels and x for metres are aligned
    cx_metres = cx * x_wrt_x + cy * x_wrt_y + a_metres[0]
    cy_metres = cx * y_wrt_x + cy * y_wrt_y + a_metres[1]

    centroids_metres.append([cx_metres, cy_metres])

  return centroids_metres


# ----- scripting to initialise and run node ----- #

if __name__ == "__main__":

  # initilise ros
  rospy.init_node("rl_localisation_node")
  rospy.loginfo("rl_localisation_node has now started")
  
  # what namespace will we use in this node for publishers/services
  node_ns = "rl" # gripper/rl

  # subscribers for image topics
  rospy.Subscriber("/camera/rgb", Image, rgb_callback)
  rospy.Subscriber("/camera/depth", Image, depth_callback)
  rospy.Subscriber("/camera/connected", Bool, connected_callback)

  # publishers for localisation data
  localisation_pub = rospy.Publisher(f"{node_ns}/objects", object_positions, queue_size=10)

  mask_pub = rospy.Publisher(f"/{node_ns}/mask", Image, queue_size=10)
  detect_pub = rospy.Publisher(f"/{node_ns}/detections", Image, queue_size=10)

  r = rospy.Rate(10) # 5hz 

  while not rospy.is_shutdown():

    if depth_camera_connected:

      rgb, depth = get_depth_image()
      depth_mask = standard_depth_process(depth)
      mask_msg = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
      mask_pub.publish(mask_msg)

      detected, centroids = gptfcn(depth_mask)
      detect_msg = bridge.cv2_to_imgmsg(detected, encoding="rgb8")
      detect_pub.publish(detect_msg)

      object_msg = object_positions()
      centroids = pixels_to_metres(centroids)

      # add all objects localised to the message
      for i, p in enumerate(centroids):
        
        # add the points
        new_point = Point()
        new_point.x = p[0]
        new_point.y = p[1]
        object_msg.xyz.append(new_point)

      # publish the object localisation message
      localisation_pub.publish(object_msg)

    r.sleep()