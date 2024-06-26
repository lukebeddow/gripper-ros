#!/home/luke/pyenv/py38_ros/bin/python

# general imports
import time
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt

# global variables
rgb_image = None
depth_image = None
camera = True
depth_camera_connected = False
xyz_points = []
object_names = []
object_confidences = []

path = "/home/luke/Documents/object_detection/depth/two_mangos_{}.png"

def depth_callback(msg):
  """
  Save the most recent depth image
  """

  im = cv2.imread(path.format(1), cv2.COLOR_BGR2RGB)

  print(im.shape)

  return im[:, :, 0]

def apply_threshold(img, lower=100, upper=240):
  img[img > upper] = 0
  img[img < lower] = 0
  return img

def remove_walls(img):
  img[:, :100] = 0
  img[:, 550:] = 0
  img[:50, :] = 0
  img[330:, :] = 0
  return img

def blob_detector(img):

  # Create SimpleBlobDetector object with default parameters
  params = cv2.SimpleBlobDetector_Params()

  # Set up the detector parameters
  params.filterByArea = True
  params.minArea = 1000
  params.maxArea = 1000000
  params.filterByCircularity = False
  params.filterByConvexity = False
  params.filterByInertia = False

  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector_create(params)

  # Detect blobs using the detector
  keypoints = detector.detect(img)

  # Draw detected blobs as red circles
  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # Display the image with the detected blobs
  cv2.imshow('Blob Detection', img_with_keypoints)

  # Wait for a key press and then exit
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def gptfcn(i=1):

  # Step 1: Read the image
  image = cv2.imread(path.format(i), cv2.IMREAD_GRAYSCALE)

  depth = depth_callback(None)
  depth = apply_threshold(depth)
  binary_image = remove_walls(depth)

  # # Step 2: Threshold the image to create a binary image
  # _, binary_image = cv2.threshold(image, 100, 240, cv2.THRESH_BINARY)

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
  output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  for centroid in centroids:
      cv2.circle(output_image, centroid, 5, (0, 0, 255), -1)

  # Display the output image with centroids
  plt.figure(figsize=(10, 10))
  plt.imshow(output_image)
  plt.title('Centroids of Blobs')
  plt.axis('off')
  plt.show()

# depth = depth_callback(None)
# depth = apply_threshold(depth)
# depth = remove_walls(depth)
# blob_detector(depth)

gptfcn()

exit()

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

  r = rospy.Rate(1) # 5hz 

  exit()

  while not rospy.is_shutdown():

    if depth_camera_connected:

      localise_objects()

      object_msg = object_positions()

      # add all objects localised to the message
      for i, p in enumerate(xyz_points):
        
        # add the points
        new_point = Point()
        new_point.x = p[0]
        new_point.y = p[1]
        new_point.z = p[2]
        object_msg.xyz.append(new_point)

        # add the object name (from yolo)
        new_name = String(object_names[i])
        object_msg.names.append(new_name)

        # add detection confidence
        new_confidence = Float32(object_confidences[i])
        object_msg.confidence.append(new_confidence)

      # publish the object localisation message
      localisation_pub.publish(object_msg)

    r.sleep()