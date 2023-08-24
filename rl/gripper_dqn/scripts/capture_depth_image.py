#!/usr/bin/env python3

## License: Apache 2.0. See LICENSE file in root directory.
## Parts of this code are
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

##################################################
##      configurable realsense viewer           ##
##################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import time

#
# NOTE: it appears that imu, rgb and depth cannot all be running simultaneously.
#       Any two of those 3 are fine, but not all three: causes timeout on wait_for_frames()
#
device_id = "128422272085"  # "923322071108" # serial number of device to use or None to use default
enable_imu = False
enable_rgb = True
enable_depth = True
# TODO: enable_pose
# TODO: enable_ir_stereo

framerate = 15
width = 640 #848
height = 360


# Configure streams
if enable_imu:
    imu_pipeline = rs.pipeline()
    imu_config = rs.config()
    if None != device_id:
        imu_config.enable_device(device_id)
    imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63) # acceleration
    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope
    imu_profile = imu_pipeline.start(imu_config)


if enable_depth or enable_rgb:
    pipeline = rs.pipeline()
    config = rs.config()

    # if we are provided with a specific device, then enable it
    if None != device_id:
        config.enable_device(device_id)

    if enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, framerate)  # depth

    if enable_rgb:
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
    if enable_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)
        if enable_depth:
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            align = rs.align(align_to)

# try:
#     frame_count = 0
#     start_time = time.time()
#     frame_time = start_time
#     while True:
#         last_time = frame_time
#         frame_time = time.time() - start_time
#         frame_count += 1

frame_count = 0
start_time = time.time()
frame_time = start_time

# last_time = frame_time
# frame_time = time.time() - start_time
# frame_count += 1

# mp_return = {"rgb" : None, "depth" : None}

# def get_depth_image(arg1=None, arg2=None, arg3=None):
#     """
#     Multiprocessing wrapper for get_depth_image
#     """

#     print("Entered get_depth_image() function")

#     queue = mp.Queue()
#     queue.put(mp_return)
    

#     process = mp.Process(target=get_depth_image_inside, args=(queue, None, None))
#     process.start()
#     process.join()

#     data = queue.get()

#     print("About to leave get_depth_image_function")

#     return (data["rgb"], data["depth"])

def get_depth_image(arg1=None, arg2=None, arg3=None):
  """
  code from: https://github.com/IntelRealSense/librealsense/issues/5628#issuecomment-575943238
  """

  global frame_count

  wait_ms = 1000

  if enable_rgb or enable_depth:
      frames = pipeline.wait_for_frames(wait_ms if (frame_count > 1) else 10000) # wait 10 seconds for first frame

  if enable_imu:
      imu_frames = imu_pipeline.wait_for_frames(wait_ms if (frame_count > 1) else 10000)

  if enable_rgb or enable_depth:
      # Align the depth frame to color frame
      aligned_frames = align.process(frames) if enable_depth and enable_rgb else None
      depth_frame = aligned_frames.get_depth_frame() if aligned_frames is not None else frames.get_depth_frame()
      color_frame = aligned_frames.get_color_frame() if aligned_frames is not None else frames.get_color_frame()

      # Convert images to numpy arrays
      depth_image = np.asanyarray(depth_frame.get_data()) if enable_depth else None
      color_image = np.asanyarray(color_frame.get_data()) if enable_rgb else None

  frame_count += 1

  return color_image, depth_image

#             # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#             depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) if enable_depth else None

#             # Stack both images horizontally
#             images = None
#             if enable_rgb:
#                 images = np.hstack((color_image, depth_colormap)) if enable_depth else color_image
#             elif enable_depth:
#                 images = depth_colormap

#             # Show images
#             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#             if images is not None:
#                 cv2.imshow('RealSense', images)

#         if enable_imu:
#             accel_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
#             gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
#             print("imu frame {} in {} seconds: \n\taccel = {}, \n\tgyro = {}".format(str(frame_count), str(frame_time - last_time), str(accel_frame.as_motion_frame().get_motion_data()), str(gyro_frame.as_motion_frame().get_motion_data())))

#         # Press esc or 'q' to close the image window
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q') or key == 27:
#             cv2.destroyAllWindows()
#             break

# finally:
#     # Stop streaming
#     pipeline.stop()

# import numpy as np
# import pyrealsense2 as rs
# from matplotlib import pyplot as plt
# from time import sleep
# import pickle

# pipeline = rs.pipeline()
# profile = pipeline.start()
# config = rs.config()

# framerate = 15 # was 30, try 15 to see if it fixes image freezing issue
# image_width = 848
# image_height = 480

# config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, framerate)	
# config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, framerate)	

# ## Get Image and Depth	
# def get_depth_image(aligning_depth_to_color=True,
#                        aligning_pcd_to_color = True,
#                        use_filtering = True):
#     # Get device product line for setting a supporting resolution

#     # Create an align object
#     # rs.align allows us to perform alignment of depth frames to others frames
#     # The "align_to" is the stream type to which we plan to align depth frames.
#     align_to = rs.stream.color
#     align = rs.align(align_to)

#     # Get frameset of color and depth
#     frames = pipeline.wait_for_frames()
#     # frames.get_depth_frame() is a 640x360 depth image

#     # Align the depth frame to color frame
#     if not aligning_depth_to_color:
#         aligned_frames = frames
#     else:
#         aligned_frames = align.process(frames)

#     # Get aligned frames
#     aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
#     color_frame = aligned_frames.get_color_frame()
#     # # Processing blocks
#     # pc = rs.pointcloud()
#     # decimate = rs.decimation_filter()
#     # decimate.set_option(rs.option.filter_magnitude, 1)
#     # aligned_depth_frame = decimate.process(aligned_depth_frame)

#     if use_filtering:
#         filters = [rs.disparity_transform(),
#                 rs.spatial_filter(),
#                 rs.temporal_filter(),
#                 rs.disparity_transform(False)]
#         for f in filters:
#             aligned_depth_frame = f.process(aligned_depth_frame)

#     depth_image = np.asanyarray(aligned_depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())

#     # if aligning_pcd_to_color:
#     #     pc.map_to(color_frame)
#     # points = pc.calculate(aligned_depth_frame)
#     # print("vert shape",np.asarray(points.get_vertices(2)).shape)
#     # h,w = color_image.shape[:2]
#     # verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
#     # tex = np.asanyarray(points.get_texture_coordinates(2))#.reshape(h, w, 3)

#     return color_image, depth_image

# if __name__ == "__main__":

#   sleep(0)

#   print("Begin image capture")

#   data = []
#   i = 0

#   num_images = 1
#   sleep_for = 0.5

#   while i < num_images:

#     sleep(sleep_for)
#     col, dep = get_depth_image(True, True, True)
#     data.append([col, dep])
#     i += 1

#   # filename = "mymujoco/jlaw_data.pickle"

#   fig, axs = plt.subplots(4, 5)

#   for j in range(num_images):
#     ix = j // 5
#     iy = j % 5
#     axs[ix][iy].imshow(data[j][0])

#   plt.imshow(dep)
#   plt.show()

#   # with open(filename, 'wb') as f:
#   #   pickle.dump(data, f)
