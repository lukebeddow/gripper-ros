#!/home/luke/pyenv/py38_ros/bin/python

# general imports
import time
import os
import torch
import numpy as np
from cv_bridge import CvBridge
import cv2
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from calibration_data import rx, ry, px, py
from statistics import median

# ros imports
import rospy
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from gripper_dqn.msg import object_positions
from gripper_dqn.srv import BinPickAuto, BinPickAutoResponse

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

# wall_x_pixel_low = 105
# wall_x_pixel_high = 545
# wall_y_pixel_low = 55
# wall_y_pixel_high = 320

# thresholds for removing floor and outliers from depth
low_threshold = 10
up_threshold = 15

# bin corner positions (d is not used)
a_metres = (0.204, 0.013)
b_metres = (0.211, 0.357)
c_metres = (0.742, 0.009)
d_metres = (0.749, 0.358)

# create openCV bridge
bridge = CvBridge()

pixel_coords = np.column_stack((px, py))

# Combine real-world coordinates into a single array
real_coords = np.column_stack((rx, ry))

# Create a Delaunay triangulation of the pixel coordinates
delaunay = Delaunay(pixel_coords)

# Create a linear interpolator based on the Delaunay triangulation
linear_interpolator = LinearNDInterpolator(delaunay, real_coords)

# Create a nearest-neighbor interpolator for extrapolation
nearest_interpolator = NearestNDInterpolator(pixel_coords, real_coords)

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
  img[img > 1] = 1

  img = np.array(img, dtype=np.uint8)

  return img

def get_centroids(binary_image, object="punnet"):

  # Step 3: Find connected components
  num_labels, labels_im = cv2.connectedComponents(binary_image)

  # Prepare to filter small components
  min_size = 3000  # Adjust this value based on the expected size of your blobs
  max_area = 10000000
  centroids = []
  iqr_range = [10, 90]

  # hand tuned values for specific objects
  if object == "punnet":
    obj_area = 18000
    obj_x = int(110 * 0.9)
    obj_y = int(160 * 0.8)
    max_area = obj_area * 1.2
  elif object == "salad":
    obj_area = 18000
    obj_x = int(170 * 0.8)
    obj_y = int(150 * 0.8)
    max_area = obj_area * 1.2
  elif object == "cucumber":
    obj_area = 8000
    obj_x = int(40 * 0.8)
    obj_y = int(260 * 0.8)
    max_area = obj_area * 1.2
  elif object == "limes":
    obj_area = 10000
    obj_x = int(110 * 0.8)
    obj_y = int(110 * 0.8)
    max_area = obj_area * 1.2
  elif object == "mango":
    obj_area = 6500
    obj_x = int(100 * 0.7)
    obj_y = int(100 * 0.7)
    max_area = obj_area * 1.2

  # Step 4: Filter out small components and calculate centroids
  for label in range(1, num_labels):  # Skipping the background label (0)
      component_mask = (labels_im == label)
      area = np.sum(component_mask)
      # print("Area is: ", area)

      if area >= min_size and area <= max_area:
            M = cv2.moments(component_mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
      elif area > max_area:
        # Step 5: Split large components based on bounding box size
        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
                # Extract all points in the contour
                points = contours[0].reshape(-1, 2)
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                
                # Calculate the interquartile range (IQR) for x and y coordinates
                Q1_x, Q3_x = np.percentile(x_coords, iqr_range)
                IQR_x = Q3_x - Q1_x
                lower_bound_x = Q1_x - 1.5 * IQR_x
                upper_bound_x = Q3_x + 1.5 * IQR_x
                
                Q1_y, Q3_y = np.percentile(y_coords, iqr_range)
                IQR_y = Q3_y - Q1_y
                lower_bound_y = Q1_y - 1.5 * IQR_y
                upper_bound_y = Q3_y + 1.5 * IQR_y
                
                # Filter out outliers
                filtered_points = points[
                    (x_coords >= lower_bound_x) & (x_coords <= upper_bound_x) &
                    (y_coords >= lower_bound_y) & (y_coords <= upper_bound_y)
                ]
                
                if filtered_points.size > 0:
                    x, y, w, h = cv2.boundingRect(filtered_points)
                    num_objects_x = max(1, w // obj_x)
                    num_objects_y = max(1, h // obj_y)
                    
                    step_x = w // num_objects_x
                    step_y = h // num_objects_y
                    
                    for i in range(num_objects_x):
                        for j in range(num_objects_y):
                            sub_x = x + i * step_x + step_x // 2
                            sub_y = y + j * step_y + step_y // 2
                            centroids.append((sub_x, sub_y))

  # Step 5: Plot centroids on the original image
  output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
  output_image *= 255 # increase contrast (prev uint8 -> 1*255=255)
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

    use_orginial = False
    if use_orginial:

      # move pixels to start at 0, 0
      cx -= wall_x_pixel_low
      cy -= wall_y_pixel_low

      # convert to factor [0.0, 1.0] of workspace
      cx /= (wall_x_pixel_high - wall_x_pixel_low)
      cy /= (wall_y_pixel_high - wall_y_pixel_low)

      # flip because depth image has 0,0 away from panda
      cx = 1.0 - cx
      # cy = 1.0 - cy # y appears in alignment

      # note x for pixels and x for metres are aligned
      cx_metres = cx * x_wrt_x + cy * x_wrt_y + a_metres[0]
      cy_metres = cx * y_wrt_x + cy * y_wrt_y + a_metres[1]

    else:

      cx_metres, cy_metres = map_pixel_to_real(cx, cy)

    centroids_metres.append([cx_metres, cy_metres])

  return centroids_metres

def map_pixel_to_real(px_val, py_val):
  """
  Map a pixel coordinate to a real-world coordinate using the interpolator.

  Parameters:
  px_val (float): x-coordinate in pixel space.
  py_val (float): y-coordinate in pixel space.

  Returns:
  tuple: Interpolated (rx, ry) real-world coordinates.
  """
  real_coord = linear_interpolator(px_val, py_val)
  
  # If the point is outside the convex hull, use the nearest-neighbor interpolator
  if np.any(np.isnan(real_coord)):
      real_coord = nearest_interpolator(px_val, py_val)

  # print(f"Pixel coordinates ({px_val}, {py_val}) map to real-world coordinates ({real_coord[0]}, {real_coord[1]})")
  
  return real_coord

def bin_pick_callback(request):

  object_msg = BinPickAutoResponse()

  num_frames = 10
  frames_done = 0

  # how many frames should a feature occur in to be included
  frame_threshold = 5

  summed_frames = None

  r = rospy.Rate(20) # 5hz 

  while not rospy.is_shutdown():

    if frames_done >= num_frames: break

    # get and process the image
    rgb, depth = get_depth_image()
    depth_mask = np.array(standard_depth_process(depth), dtype=np.uint8)

    # sum all frames
    if summed_frames is None:
       summed_frames = depth_mask
    else:
       summed_frames += depth_mask

    frames_done += 1
    r.sleep()

  # apply a threshold on the frames and return to binary mask
  summed_frames[summed_frames < frame_threshold] = 0
  summed_frames[summed_frames >= frame_threshold] = 1
  final_mask = np.array(summed_frames, dtype=np.uint8)

  # now extract the centroids (and publish final frame)
  detected, centroids = get_centroids(final_mask, object=request.object_name)
  centroid_msg = bridge.cv2_to_imgmsg(detected, encoding="rgb8")
  centroid_pub.publish(centroid_msg)

  # for videos: save the detection image
  path = "/home/luke/Documents/object_detection/video_detections/"
  name_format = "{0}_detection_{1}.png"
  rgb_format = "{0}_rgb_{1}.png"
  tries = 50
  for t in range(tries):
    this_name = path + name_format.format(request.object_name, t+1)
    if not os.path.exists(this_name):
      cv2.imwrite(this_name, detected)
      cv2.imwrite(path + rgb_format.format(request.object_name, t+1), 
                  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
      break

  # create a centroid message, and fill it in
  response = BinPickAutoResponse()
  centroids = pixels_to_metres(centroids)

  points = []

  # add all objects localised to the message
  for i, p in enumerate(centroids):
    
    # add the points
    new_point = Point()
    new_point.x = p[0]
    new_point.y = p[1]
    response.centroids.append(new_point)

  return response

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
  centroid_pub = rospy.Publisher(f"/{node_ns}/centroids", Image, queue_size=10)

  # service for object positions on request
  rospy.Service(f"/{node_ns}/get_object_centroids", BinPickAuto, bin_pick_callback)

  r = rospy.Rate(10) # 5hz 

  while not rospy.is_shutdown():

    if depth_camera_connected:
       
      pass

      # rgb, depth = get_depth_image()
      # depth_mask = standard_depth_process(depth)
      # mask_msg = bridge.cv2_to_imgmsg(depth, encoding="32FC1")
      # mask_pub.publish(mask_msg)

      # detected, centroids = get_centroids(depth_mask)
      # detect_msg = bridge.cv2_to_imgmsg(detected, encoding="rgb8")
      # detect_pub.publish(detect_msg)

      # object_msg = object_positions()
      # centroids = pixels_to_metres(centroids)

      # # add all objects localised to the message
      # for i, p in enumerate(centroids):
        
      #   # add the points
      #   new_point = Point()
      #   new_point.x = p[0]
      #   new_point.y = p[1]
      #   object_msg.xyz.append(new_point)

      # # publish the object localisation message
      # localisation_pub.publish(object_msg)

    r.sleep()