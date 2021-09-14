#include <segmented_controller/segmented_control.h>
#include <pluginlib/class_list_macros.h>

namespace segmented_controller
{
    typedef joint_trajectory_controller::JointTrajectoryController<trajectory_interface::QuinticSplineSegment<double>,
            hardware_interface::EffortJointInterface> JointTrajectoryController;

} // namespace

PLUGINLIB_EXPORT_CLASS(segmented_controller::JointTrajectoryController, controller_interface::ControllerBase);

