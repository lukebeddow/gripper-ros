#pragma once

#include <ros/ros.h>
#include <gripper_msgs/pose.h>

class Gripper
{
  /* a class for representing the gripper state all units SI unless stated. Fingers
  tilting inwards is a +ve angle, outwards is a -ve angle */

public:

  constexpr static double leadscrew_dist = 35e-3;
  constexpr static double to_rad = 3.1415926535897 / 180.0;
  constexpr static double to_deg = 180.0 / 3.1415926535897;

  // value limits - these should be exact but then will include a small tolerance
  constexpr static double xy_min =  50e-3;
  constexpr static double z_min = 0e-3;
  constexpr static double xy_max = 140e-3;
  constexpr static double z_max = 160e-3;
  constexpr static double th_min = -40 * to_rad;
  constexpr static double th_max = 40 * to_rad;
  constexpr static double tol = 1e-4;

  // motor positions in meters
  double x_;
  double y_;
  double z_;

  // constructor
  Gripper() : x_(xy_max), y_(xy_max), z_(z_min) {}; // default position
  Gripper(double x, double y, double z) {
    // initialise
    x_ = x; y_ = y; z_ = z;
    check_limits();
  }

  // convert y position to and from angle
  double calc_y(double th_rad) { return x_ + leadscrew_dist * sin(th_rad); }
  double calc_th(double x, double y) { return atan((x - y) / leadscrew_dist); }

  // basic setting functions for meters, millimeters, degrees and radians
  void set_xy_m(double x_m, double y_m) { 
    x_ = x_m; y_ = y_m; check_xy(); 
  }
  void set_xy_mm(double x_mm, double y_mm) { 
    x_ = x_mm * 1e-3; y_ = y_mm * 1e-3; check_xy(); 
  }
  void set_xy_m_rad(double x_m, double th_rad) {
    x_ = x_m; set_th_rad(th_rad); check_xy(); 
  }
  void set_xy_mm_rad(double x_mm, double th_rad) {
    x_ = x_mm * 1e-3; set_th_rad(th_rad); check_xy(); 
  }
  void set_xy_m_deg(double x_m, double th_deg) {
    x_ = x_m; set_th_deg(th_deg); check_xy(); 
  }
  void set_xy_mm_deg(double x_mm, double th_deg) {
    x_ = x_mm * 1e-3; set_th_deg(th_deg); check_xy(); 
  }
  void set_z_m(double z_m) { z_ = z_m; check_z(); }
  void set_z_mm(double z_mm) { z_ = z_mm * 1e-3; check_z(); }
  void set_th_rad(double th_rad) { y_ = calc_y(th_rad); check_xy(); }
  void set_th_deg(double th_deg) { set_th_rad(th_deg * to_rad); check_xy(); }

  // getter functions for the same units
  double get_x_m() { return x_; }
  double get_y_m() { return y_; }
  double get_z_m() { return z_; }
  double get_x_mm() { return x_ * 1e3; }
  double get_y_mm() { return y_ * 1e3; }
  double get_z_mm() { return z_ * 1e3; }
  double get_th_rad() { return calc_th(x_, y_); }
  double get_th_deg() { return to_deg * get_th_rad(); }
  double get_prismatic_joint() { return x_; }
  double get_revolute_joint() { return get_th_rad() * -1; } // differing conventions
  double get_palm_joint() { return z_; }

  // member function declarations
  void print();
  void check_z();
  void check_xy();
  void check_limits();
  Gripper from_msg(gripper_msgs::pose);
  gripper_msgs::pose to_msg();
  void add_msg(gripper_msgs::pose nudge);

};

void Gripper::check_xy() 
{
  // first check leadscrew physical limits for x, and enforce these
  if (x_ > xy_max + tol) {
    ROS_WARN("Gripper received out of range (too large) value x, capped at max");
    x_ = xy_max;
  }
  if (x_ < xy_min - tol) {
    ROS_WARN("Gripper received out of range (too small) value x, capped at min");
    x_ = xy_min;
  }

  // check leadscrew physical limits for y as well
  if (y_ > xy_max + tol) {
    ROS_WARN("Gripper received out of range (too large) value y, capped at max");
    y_ = xy_max;
  }
  if (y_ < xy_min - tol) {
    ROS_WARN("Gripper received out of range (too small) value y, capped at min");
    y_ = xy_min;
  }

  // now check that we don't result in an out of bounds angle
  double th = calc_th(x_, y_);
  if (th > th_max) {
    ROS_WARN("Gripper received xy values that exceeds angle limits, capped at max");
    y_ = calc_y(-1 * th_max); // note the -1 to get desired behaviour
  }
  if (th < th_min) {
    ROS_WARN("Gripper received xy values that exceeds angle limits, capped at max");
    y_ = calc_y(-1 * th_min);
  }
}

void Gripper::check_z() 
{
  // check leadscrew physical limits, and enforce these
  if (z_ > z_max + tol) {
    ROS_WARN("Gripper received out of range (too large) value z, capped at max");
    z_ = z_max;
  }
  if (z_ < z_min - tol) {
    ROS_WARN("Gripper received out of range (too small) value z, capped at min");
    z_ = z_min;
  }
}

void Gripper::check_limits()
{
  /* This function checks that all motor values constitute a safe and legal
  position, if not, these limits are enforced */

  check_xy();
  check_z();
}

void Gripper::print()
{
  /* print the gripper state in ros */

  char indent[] = "\t\t";

  ROS_INFO_STREAM("The gripper state in mm:\n"
    << indent << "x = " << get_x_mm() << "\n"
    << indent << "y = " << get_y_mm() << "\n"
    << indent << "z = " << get_z_mm() << "\n"
    << indent << "th = " << get_th_rad() << " (" << get_th_deg() << " degrees)");
}

Gripper Gripper::from_msg(gripper_msgs::pose pose)
{
  /* This converts a gripper message into a gripper object */

  return Gripper(pose.x, pose.y, pose.z);
}

gripper_msgs::pose Gripper::to_msg()
{
  /* this outputs a gripper message of the current pose */

  gripper_msgs::pose pose;
  pose.x = x_;
  pose.y = y_;
  pose.z = z_;

  return pose;
}

void Gripper::add_msg(gripper_msgs::pose nudge)
{
  /* this adds the contents of a gripper_msgs/pose to the gripper */

  x_ += nudge.x;
  y_ += nudge.y;
  z_ += nudge.z;

  check_limits();
}
