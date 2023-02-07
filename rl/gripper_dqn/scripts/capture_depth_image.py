#!/usr/bin/env python3

import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from time import sleep
import pickle

pipeline = rs.pipeline()
profile = pipeline.start()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)	
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)	

## Get Image and Depth	
def get_depth_image(aligning_depth_to_color=True,
                       aligning_pcd_to_color = True,
                       use_filtering = True):
    # Get device product line for setting a supporting resolution

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    if not aligning_depth_to_color:
        aligned_frames = frames
    else:
        aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    # # Processing blocks
    # pc = rs.pointcloud()
    # decimate = rs.decimation_filter()
    # decimate.set_option(rs.option.filter_magnitude, 1)
    # aligned_depth_frame = decimate.process(aligned_depth_frame)

    if use_filtering:
        filters = [rs.disparity_transform(),
                rs.spatial_filter(),
                rs.temporal_filter(),
                rs.disparity_transform(False)]
        for f in filters:
            aligned_depth_frame = f.process(aligned_depth_frame)

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # if aligning_pcd_to_color:
    #     pc.map_to(color_frame)
    # points = pc.calculate(aligned_depth_frame)
    # print("vert shape",np.asarray(points.get_vertices(2)).shape)
    # h,w = color_image.shape[:2]
    # verts = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
    # tex = np.asanyarray(points.get_texture_coordinates(2))#.reshape(h, w, 3)

    return color_image, depth_image

if __name__ == "__main__":

  sleep(0)

  print("Begin image capture")

  data = []
  i = 0

  num_images = 20
  sleep_for = 0.5

  while i < num_images:

    sleep(sleep_for)
    col, dep = get_depth_image(True, True, True)
    data.append([col, dep])
    i += 1

  filename = "mymujoco/jlaw_data.pickle"

  fig, axs = plt.subplots(4, 5)

  for j in range(num_images):
    ix = j // 5
    iy = j % 5
    axs[ix][iy].imshow(data[j][0])

  # plt.imshow(dep)
  plt.show()

  with open(filename, 'wb') as f:
    pickle.dump(data, f)
