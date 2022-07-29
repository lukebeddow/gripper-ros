#include "gripper.h"

namespace luke
{

bool Gripper::update_xy() 
{
  /* checks that the x and y values are within bounds, and also updates the
  internal state of the angle, th, and motor step counts */

  bool within_limits = true;

  // first check leadscrew physical limits for x, and enforce these
  if (x > xy_max + limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too large) value x = " 
        << x << ", capped at max = " << xy_max << "\n";
    }
    x = xy_max;
    within_limits = false;
  }
  if (x < xy_min - limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too small) value x = "
        << x << ", capped at min = " << xy_min << "\n";
    }
    x = xy_min;
    within_limits = false;
  }

  // check leadscrew physical limits for y as well
  if (y > xy_max + limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too large) value y = "
        << y << ", capped at max = " << xy_max << "\n";
    }
    y = xy_max;
    within_limits = false;
  }
  if (y < xy_min - limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too small) value y = "
        << y << ", capped at min = " << xy_min << "\n";
    }
    y = xy_min;
    within_limits = false;
  }

  // now check that we don't result in an out of bounds angle
  double new_th = calc_th(x, y);
  if (new_th > th_max) {
    if (debug) {
      std::cout << "Gripper received xy values that exceeds angle limits = " 
        << new_th << ", capped at max = " << th_max << "\n";
    }
    new_th = th_max;
    y = calc_y(th_max);
    within_limits = false;
  }
  if (new_th < th_min) {
    if (debug) {
      std::cout << "Gripper received xy values that exceeds angle limits = "
        << new_th << ", capped at min = " << th_min << "\n";
    }
    new_th = th_min;
    y = calc_y(th_min);
    within_limits = false;
  }

  // update our saved record of the finger angle
  th = new_th;

  // finally, check that the fingertips are not overlapping too much
  double th_lim = calc_max_fingertip_angle();
  if (th < th_lim) {
    if (debug) {
      std::cout << "Gripper received xy values that exceed fingertip radius limit = "
        << fingertip_radius_min << ", angle capped at = " << th_lim << "\n";
    }
    y = calc_y(th_lim);
    th = th_lim;
    within_limits = false;
  } 

  // update the motor step counts
  step.x = get_x_step();
  step.y = get_y_step();

  return within_limits;
}

bool Gripper::update_z() 
{
  bool within_limits = true;

  // check leadscrew physical limits, and update motor step count
  if (z > z_max + limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too large) value z"
        ", capped at max = " << z_max << "\n";
    }
    z = z_max;
    within_limits = false;
  }
  if (z < z_min - limit_tol) {
    if (debug) {
      std::cout << "Gripper received out of range (too small) value z"
        ", capped at min = " << z_min << "\n";
    }
    z = z_min;
    within_limits = false;
  }

  // update the motor step counts
  step.z = get_z_step();

  return within_limits;
}

bool Gripper::update()
{
  /* This function checks that the state is within safe limits and also updates
  the theta and step counters */

  bool within_limits = true;

  within_limits *= update_xy();
  within_limits *= update_z();

  return within_limits;
}

bool Gripper::update_x_th_z()
{
  /* special function to update the gripper state based on the assumption that
  the x, th, and z values have been set, and nothing else. The y value is
  completely overriden in this case by the theta value */

  // override y with our theta value
  y = calc_y(th);

  return update();
}

void Gripper::print()
{
  /* print the gripper state */

  std::printf("The gripper state is:\n\t"
    "x = %.2f mm (%i steps)\n\t"
    "y = %.2f mm (%i steps)\n\t"
    "z = %.2f mm (%i steps)\n\t"
    "th = %.2f rad (%.2f deg)\n", get_x_mm(), step.x, get_y_mm(), step.y, 
      get_z_mm(), step.z, get_th_rad(), get_th_deg());

}

bool Gripper::step_to(double xstep, double ystep, double zstep, int num)
{
  /* steps each motor num steps towards the desired step state */

  bool finished = true;

  // calculate distance to target position
  int x_to_go = xstep - step.x;
  int y_to_go = ystep - step.y;
  int z_to_go = zstep - step.z;

  // how many steps will we go
  if (x_to_go < 0) {
    if (x_to_go * -1 > num) {
      x_to_go = -1 * num;
      finished = false;
    }
  }
  else {
    if (x_to_go > num) {
      x_to_go = num;
      finished = false;
    }
  }

  if (y_to_go < 0) {
    if (y_to_go * -1 > num) {
      y_to_go = -1 * num;
      finished = false;
    }
  }
  else {
    if (y_to_go > num) {
      y_to_go = num;
      finished = false;
    }
  }

  if (z_to_go < 0) {
    if (z_to_go * -1 > num) {
      z_to_go = -1 * num;
      finished = false;
    }
  }
  else {
    if (z_to_go > num) {
      z_to_go = num;
      finished = false;
    }
  }
  
  // set the new position
  set_xyz_step(step.x + x_to_go, step.y + y_to_go, step.z + z_to_go);

  return finished;
}

bool Gripper::step_to(Gripper target, int num)
{
  /* steps towards a given gripper state */

  /* old code
  for (int i = 0; i < num; i++) {
    if (step_to(target.step.x, target.step.y, target.step.z)) {
      return true;
    }
  }
  */

  return step_to(target.step.x, target.step.y, target.step.z, num);
}

bool Gripper::step_to_m_rad(double x, double th, double z)
{
  /* overload for making only a single step at a time */

  return step_to_m_rad(x, th, z, 1);
}

bool Gripper::step_to_m_rad(double x, double th, double z, int num)
{
  /* steps each motor one step towards the desired state */

  // std::printf("step_to_m_rad has been asked for (x, th, z) of (%.3f, %.3f, %.3f)\n",
  //   x, th, z);

  // get the steps of the desired state
  Gripper temp;
  temp.set_xyz_m_rad(x, th, z);

  return step_to(temp, num);
}

bool Gripper::is_at_xyz_m(double x, double y, double z)
{
  /* is the gripper at a target motor position state */

  Gripper target;
  target.set_xyz_m(x, y, z);

  return is_at(target);
}

bool Gripper::is_at_xyz_m_rad(double x, double th, double z)
{
  /* is the gripper at the target joint state */

  Gripper target;
  target.set_xyz_m_rad(x, th, z);

  return is_at(target);
}

bool Gripper::is_at_step(int xstep, int ystep, int zstep)
{
  /* is the gripper at the target step state */

  Gripper target;
  target.set_xyz_step(xstep, ystep, zstep);

  return is_at(target);
}

bool Gripper::is_at(Gripper target)
{
  /* Is the gripper at a target step state */

  update();

  if (step.x == target.step.x and
      step.y == target.step.y and 
      step.z == target.step.z)
      return true;

  return false;
}

bool Gripper::is_at_xyz_m(double x, double y, double z, int tol)
{
  /* is the gripper at a target motor position state with tol no. of steps */

  Gripper target;
  target.set_xyz_m(x, y, z);

  return is_at(target, tol);
}

bool Gripper::is_at_xyz_m_rad(double x, double th, double z, int tol)
{
  /* is the gripper at the target joint state within tol no. of steps */

  Gripper target;
  target.set_xyz_m_rad(x, th, z);

  return is_at(target, tol);
}

bool Gripper::is_at_step(int xstep, int ystep, int zstep, int tol)
{
  /* is the gripper at the target step state within tol no. of steps */

  Gripper target;
  target.set_xyz_step(xstep, ystep, zstep);

  return is_at(target, tol);
}

bool Gripper::is_at(Gripper target, int tol)
{
  /* is the gripper within the tolerance of the target step state */

  update();

  // std::cout << "bools are: " << (abs(step.x - target.step.x) <= tol)
  //   << ", " << (abs(step.y - target.step.y) <= tol) 
  //   << ", " << (abs(step.z - target.step.z) <= tol) << '\n';

  if (abs(step.x - target.step.x) <= tol and
      abs(step.y - target.step.y) <= tol and
      abs(step.z - target.step.z) <= tol)
      return true;
  
  return false;
}

} // namespace luke