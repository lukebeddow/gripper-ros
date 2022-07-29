#ifndef GRIPPER_H_
#define GRIPPER_H_

#include <iostream>
#include <cstdio>
#include <math.h>

namespace luke
{

class Gripper
{
  /* class representing the gripper state, all units SI unless stated */

public:

  /* ----- Member variables ------ */

  // finger angle sign convention, +1 means inward -ve, -1 for inward +ve
  /* the urdf uses inward -ve, the real gripper uses inward +ve */
  constexpr static int sign = 1;

  // are we in debug mode
  constexpr static bool debug = false;

  // handy constants
  constexpr static double to_rad = M_PI / 180.0;
  constexpr static double to_deg = 180.0 / M_PI;
  constexpr static double limit_tol = 1e-4;     // for floating point limit checks

  // gripper physical constants - user set
  constexpr static double leadscrew_dist = 35e-3;
  constexpr static double finger_length = 235e-3;
  constexpr static double hook_length = 39.5e-3;

  constexpr static double xy_lead = 4;
  constexpr static double xy_gear = 1.5;
  constexpr static double xy_steps_per_rev = 400;

  constexpr static double z_lead = 4.8768;
  constexpr static double z_gear = 1;
  constexpr static double z_steps_per_rev = 400;

  // hard gripper limits - user set
  constexpr static double xy_min =  49e-3;
  constexpr static double xy_max = 134e-3;
  constexpr static double z_min = 0e-3;
  constexpr static double z_max = 165e-3;

  // home position - user set
  constexpr static double xy_home = xy_max - 1.0 * (xy_lead * 1e-3 / xy_gear); // home is 1.0rev from max
  constexpr static double z_home = z_min + 1.0 * (z_lead * 1e-3 / z_gear);     // see Gripper_v2.h (Arduino library)

  // safety limits - user set
  double th_min = -40 * to_rad;
  double th_max = 40 * to_rad;
  double xy_max_force = 20;
  double z_max_force = 10;
  double fingertip_radius_min = -5e-3;

  // auto generated constants
  constexpr static double hypotenuse = sqrt(pow(finger_length, 2) + pow(hook_length, 2));
  constexpr static double resting_angle = atan(hook_length / finger_length);
  constexpr static double xy_step_m = xy_lead / (xy_gear * xy_steps_per_rev * 1e3);
  constexpr static double z_step_m = z_lead / (z_gear * z_steps_per_rev * 1e3);

  // motor positions in metres
  double x, y, z;

  // finger angle in radians and motor step counts
  // NOTE: these are only ever updated in update() / update_xy() / update_z()
  double th;
  struct { int x, y, z; } step;

  /* ----- Member functions ----- */

  // constructor
  Gripper() : x(xy_home), y(xy_home), z(z_home) { update(); } // default position

  // reset to default
  void reset() { x = xy_home; y = xy_home; z = z_home; update(); }

  // convert y position to and from angle, take note of chosen sign convention
  double calc_y(double th_rad) { return x + sign * leadscrew_dist * sin(th_rad); }
  double calc_th(double x, double y) { return atan((y - x) / leadscrew_dist) * sign; }

  // checking fingertip radius
  double calc_fingertip_radius() { 
    return x - hypotenuse * sin(resting_angle - th * sign);
  }
  double calc_max_fingertip_angle() { 
    return (asin((fingertip_radius_min - x) / hypotenuse) + resting_angle) * sign;
  }

  // getter functions
  double get_x_m() { return x; }
  double get_y_m() { return y; }
  double get_z_m() { return z; }
  double get_x_mm() { return x * 1e3; }
  double get_y_mm() { return y * 1e3; }
  double get_z_mm() { return z * 1e3; }
  double get_th_rad() { return calc_th(x, y); }
  double get_th_deg() { return to_deg * get_th_rad(); }
  double get_prismatic_joint() { return x; }
  double get_revolute_joint() { return get_th_rad() * sign * -1; } // always inward -ve
  double get_palm_joint() { return z; }
  int get_x_step() { return round((xy_max - x) / xy_step_m); };
  int get_y_step() { return round((xy_max - y) / xy_step_m); };
  int get_z_step() { return round(z / z_step_m); };

  // setters for setting x,y,z individually
  // NB all setters return a boolean "within_limits"
  bool set_x_m(double x_m) { x = x_m; return update_xy(); }
  bool set_x_mm(double x_mm) { x = x_mm * 1e-3; return update_xy(); }
  bool set_y_m(double y_m) { y = y_m; return update_xy(); }
  bool set_y_mm(double y_mm) { y = y_mm * 1e-3; return update_xy(); }
  bool set_z_m(double z_m) { z = z_m; return update_z(); }
  bool set_z_mm(double z_mm) { z = z_mm * 1e-3; return update_z(); }
  bool set_th_rad(double th_rad) { y = calc_y(th_rad); return update_xy(); }
  bool set_th_deg(double th_deg) { return set_th_rad(th_deg * to_rad); }

  // setters for setting both x and y, or all three
  bool set_xy_m(double x_m, double y_m) { 
    x = x_m; y = y_m; return update_xy(); 
  }
  bool set_xy_mm(double x_mm, double y_mm) { 
    x = x_mm * 1e-3; y = y_mm * 1e-3; return update_xy(); 
  }
  bool set_xy_m_rad(double x_m, double th_rad) {
    x = x_m; return set_th_rad(th_rad);
  }
  bool set_xy_mm_rad(double x_mm, double th_rad) {
    x = x_mm * 1e-3; return set_th_rad(th_rad);
  }
  bool set_xy_m_deg(double x_m, double th_deg) {
    x = x_m; return set_th_deg(th_deg); 
  }
  bool set_xy_mm_deg(double x_mm, double th_deg) {
    x = x_mm * 1e-3; return set_th_deg(th_deg); 
  }
  bool set_xyz_mm(double x_mm, double y_mm, double z_mm) {
    x = x_mm * 1e-3; y = y_mm * 1e-3; z = z_mm * 1e-3; return update();
  }
  bool set_xyz_mm_deg(double x_mm, double th_deg, double z_mm) {
    x = x_mm * 1e-3; bool in_lim = set_th_deg(th_deg); z = z_mm * 1e-3;
    return (update_z() ? in_lim : false);
  }
  bool set_xyz_mm_rad(double x_mm, double th_rad, double z_mm) {
    x = x_mm * 1e-3; bool in_lim = set_th_rad(th_rad); z = z_mm * 1e-3;
    return (update_z() ? in_lim : false);
  }
  bool set_xyz_m(double x_m, double y_m, double z_m) {
    x = x_m; y = y_m; z = z_m; return update();
  }
  bool set_xyz_m_deg(double x_m, double th_deg, double z_m) {
    x = x_m; bool in_lim = set_th_deg(th_deg); z = z_m; 
    return (update_z() ? in_lim : false);
  }
  bool set_xyz_m_rad(double x_m, double th_rad, double z_m) {
    x = x_m; bool in_lim = set_th_rad(th_rad); z = z_m;
    return (update_z() ? in_lim : false);
  }
  bool set_xyz_step(int xstep, int ystep, int zstep) {
    bool in_lim = set_xy_m(xy_max - xy_step_m * xstep, xy_max - xy_step_m * ystep);
    return (set_z_m(z_step_m * zstep) ? in_lim : false);
  }

  // stepping
  double step_x(int num) { set_x_m(xy_max - xy_step_m * (step.x + num)); return x; }
  double step_y(int num) { set_y_m(xy_max - xy_step_m * (step.y + num)); return y; }
  double step_z(int num) { set_z_m(z_step_m * (step.z + num)); return z; }

  // functions defined in gripper.cpp below

  void print();

  // update theta and step values, plus returns boolean "within_limits"
  bool update_z();
  bool update_xy();
  bool update();
  bool update_x_th_z();

  // step towards a target
  bool step_to(double xstep, double ystep, double zstep, int num = 1);
  bool step_to(Gripper target, int num);
  bool step_to_m_rad(double x, double th, double z, int num);
  bool step_to_m_rad(double x, double th, double z);

  // equality check with different units and overloaded with tolerances
  bool is_at(Gripper target);
  bool is_at(Gripper target, int tol);
  bool is_at_xyz_m(double x, double y, double z);
  bool is_at_xyz_m(double x, double y, double z, int tol);
  bool is_at_xyz_m_rad(double x, double th, double z);
  bool is_at_xyz_m_rad(double x, double th, double z, int tol);
  bool is_at_step(int xstep, int ystep, int zstep);
  bool is_at_step(int xstep, int ystep, int zstep, int tol);
};

inline std::ostream& operator<<(std::ostream& os, luke::Gripper& g) {
  g.print();
  return os;
}

} // namespace luke

#endif // GRIPPER_H_
