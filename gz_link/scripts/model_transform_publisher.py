#!/usr/bin/env python

import rospy
import tf2_ros
import gazebo_msgs.msg
import geometry_msgs.msg
import time

prev_publish_time = time.time()
rate = 10

def callback(data):
  """Callback function to publish the model transforms"""

  # prepare varibles
  transform_list = []
  global last_published
  global rate
  publish_time_gap = 1.0 / rate

  # publish at specified rate
  if time.time() - prev_publish_time > publish_time_gap:
    for i in range(len(data.name)):
      # create the transform
      transform = geometry_msgs.msg.TransformStamped()
      # stamp
      transform.header.stamp = rospy.Time.now()
      transform.header.frame_id = "panda_link0"
      transform.child_frame_id = data.name[i]
      # fill in data
      transform.transform.translation.x = data.pose[i].position.x
      transform.transform.translation.y = data.pose[i].position.y
      transform.transform.translation.z = data.pose[i].position.z
      transform.transform.rotation.w = data.pose[i].orientation.w
      transform.transform.rotation.x = data.pose[i].orientation.x
      transform.transform.rotation.y = data.pose[i].orientation.y
      transform.transform.rotation.z = data.pose[i].orientation.z
      # broadcast
      transform_list.append(transform)
      broadcaster.sendTransform(transform_list)
  
  return

if __name__ == '__main__':

  rospy.init_node('gazebo_tf_broadcaster')

  broadcaster = tf2_ros.StaticTransformBroadcaster()

  rospy.Subscriber("/gazebo/model_states", gazebo_msgs.msg.ModelStates, callback)

  rospy.spin()
