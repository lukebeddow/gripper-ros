#include "ros/ros.h"
#include <kdl/jntarrayvel.hpp>
// #include <arm_control/Efforts.h>
#include <kdl_parser/kdl_parser.hpp>
#include <realtime_tools/realtime_publisher.h>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <trajectory_interface/quintic_spline_segment.h>
#include <joint_trajectory_controller/joint_trajectory_controller.h>

// added by Luke following build time error
#include <boost/scoped_ptr.hpp>

#include <segmented_controller/Efforts.h>

namespace joint_trajectory_controller
{
    typedef trajectory_interface::QuinticSplineSegment<double> SegmentImpl;
    typedef JointTrajectorySegment<SegmentImpl> Segment;
    typedef typename Segment::State State;
}

template<>
class HardwareInterfaceAdapter<hardware_interface::EffortJointInterface, joint_trajectory_controller::State>
{
public:
    HardwareInterfaceAdapter() :
            joint_handles_ptr(0)
    {}

    bool init(std::vector<hardware_interface::JointHandle> &joint_handles, ros::NodeHandle &nh)
    {
        // Store pointer to joint handles
        joint_handles_ptr = &joint_handles;

        // Parse the URDF string into a URDF model.
        urdf::Model urdf_model;
        if (!urdf_model.initParam("/robot_description")) {
            ROS_ERROR("Failed to parse urdf model from robot description");
            return false;
        }
        ROS_INFO("Parsed urdf model from robot description");

        // Compute the KDL tree of the robot from the URDF.
        KDL::Tree tree;
        if (!kdl_parser::treeFromUrdfModel(urdf_model, tree)) {
            ROS_ERROR("Failed to parse kdl tree from urdf model");
            return false;
        }
        ROS_INFO("Parsed kdl tree from urdf model");

        // Extract chain from KDL tree.
        KDL::Chain chain;

        // testing as we know we want one controller per finger
        static bool finger_1 = false;
        static bool finger_2 = false;
        static bool finger_3 = false;

        if (not finger_1) {
          if (!tree.getChain("finger_1_segment_link_1", "finger_1_finger_hook_link", chain)) {
            ROS_ERROR("Failed to extract chain from 'toe' to 'finger' in kdl tree");
            return false;
          }
          finger_1 = true;
          ROS_INFO("Extracted finger_1 chain from kdl tree");
        }
        else if (not finger_2) {
          if (!tree.getChain("finger_2_segment_link_1", "finger_2_finger_hook_link", chain)) {
            ROS_ERROR("Failed to extract chain from 'toe' to 'finger' in kdl tree");
            return false;
          }
          finger_2 = true;
          ROS_INFO("Extracted finger_2 chain from kdl tree");
        }
        else if (not finger_3) {
          if (!tree.getChain("finger_3_segment_link_1", "finger_3_finger_hook_link", chain)) {
            ROS_ERROR("Failed to extract chain from 'toe' to 'finger' in kdl tree");
            return false;
          }
          finger_3 = true;
          ROS_INFO("Extracted finger_3 chain from kdl tree");
        }
        else {
          ROS_ERROR("finger_1, 2 and 3 all = true in segmented_control.h");
        }


        // Init effort command publisher.
        publisher.reset(new EffortsPublisher(nh, "efforts", 10));

        // links joints efforts to publisher message.
        joints_efforts = &(publisher->msg_.data);

        // Reset and resize joint states/controls.
        unsigned int n_joints = chain.getNrOfJoints();
        // inner_loop_control.resize(n_joints);
        // outer_loop_control.resize(n_joints);
        joints_effort_limits.resize(n_joints);
        (*joints_efforts).resize(n_joints);
        joints_state.resize(n_joints);

        joint_positions.resize(n_joints);
        joint_velocities.resize(n_joints);
        joint_efforts.resize(n_joints);

        /*
        for (unsigned int idx = 0; idx < chain.getNrOfJoints(); idx++) {
            // Get joint name.
            std::string name = chain.getSegment(idx).getJoint().getName();

            // Extract joint effort limits from urdf.
            if (!(urdf_model.getJoint(name)) ||
                !(urdf_model.getJoint(name)->limits) ||
                !(urdf_model.getJoint(name)->limits->effort)) {
                ROS_ERROR("No effort limit specified for joint '%s'", name.c_str());
                return false;
            }
            joints_effort_limits.data[idx] = urdf_model.getJoint(name)->limits->effort;
        }
        ROS_INFO("Extracted joint effort limits");
        */

        /*
        // Init inverse dynamics solver.
        id_solver.reset(new KDL::ChainIdSolver_RNE(chain, KDL::Vector(0, 0, -9.81)));
        ROS_INFO("Initialized kdl inverse dynamics solver");
        */

        return true;
    }

    void starting(const ros::Time & /*time*/)
    {
        if (!joint_handles_ptr) { return; }

        for (unsigned int idx = 0; idx < joint_handles_ptr->size(); ++idx) {
            // Write joint effort command.
            (*joint_handles_ptr)[idx].setCommand(0);
        }
    }

    void stopping(const ros::Time & /*time*/)
    {}

    void updateCommand(const ros::Time &     /*time*/,
                       const ros::Duration & /*period*/,
                       const joint_trajectory_controller::State &desired_state,
                       const joint_trajectory_controller::State &state_error)
    {
        if (!joint_handles_ptr) { return; }

        /*
        for (size_t idx = 0; idx < joint_handles_ptr->size(); ++idx) {

            // Update joint state with current position (q) and velocity (qdot).
            joints_state.q.data[idx] = (*joint_handles_ptr)[idx].getPosition();
            joints_state.qdot.data[idx] = (*joint_handles_ptr)[idx].getVelocity();

            // Compute outer loop control.
            // todo: dynamic reconfigure parameters.
            outer_loop_control.data[idx] = 100 * state_error.position[idx] + 10 * state_error.velocity[idx];
        }

        // No external forces (except gravity).
        KDL::Wrenches external_forces(joint_handles_ptr->size());

        // Solve inverse dynamics (inner loop control).
        if (id_solver->CartToJnt(
                joints_state.q,
                joints_state.qdot,
                outer_loop_control,
                external_forces,
                inner_loop_control) != 0) {
            ROS_ERROR("error solving inverse dynamics");
            return;
        };

        for (unsigned int idx = 0; idx < joint_handles_ptr->size(); ++idx) {
            (*joints_efforts)[idx] = inner_loop_control.data[idx];

            // Limit based on min/max efforts.
            (*joints_efforts)[idx] = std::min((*joints_efforts)[idx], joints_effort_limits.data[idx]);
            (*joints_efforts)[idx] = std::max((*joints_efforts)[idx], -joints_effort_limits.data[idx]);

            // Write joint effort command.
            (*joint_handles_ptr)[idx].setCommand((*joints_efforts)[idx]);

        }
        */

        double joint_stiffness = 100.0;
        double joint_damping = 0.0;

        for (unsigned int idx = 0; idx < joint_handles_ptr->size(); ++idx) {

            // (*joints_efforts)[idx] = inner_loop_control.data[idx];

            // get joints current position and velocity data
            joint_positions[idx] = (*joint_handles_ptr)[idx].getPosition();
            joint_velocities[idx] = (*joint_handles_ptr)[idx].getVelocity();

            // calculate the control force
            joint_efforts[idx] = -joint_positions[idx] * joint_stiffness
              + joint_velocities[idx] * joint_damping;

            /*
            // Limit based on min/max efforts.
            (*joints_efforts)[idx] = std::min((*joints_efforts)[idx], joints_effort_limits.data[idx]);
            (*joints_efforts)[idx] = std::max((*joints_efforts)[idx], -joints_effort_limits.data[idx]);
            */

            // *joints_efforts is linked to the publisher msg currently
            (*joints_efforts)[idx] = joint_efforts[idx];
            
            // Write joint effort command.
            (*joint_handles_ptr)[idx].setCommand(joint_efforts[idx]);

            // ROS_INFO_STREAM("Joint " << idx << " has effort " << joint_efforts[idx]
            //   << " and position " << joint_positions[idx] << " and velocity "
            //   << joint_velocities[idx]);
        }

      


        // Publish efforts.
        if (publisher->trylock()) {
            publisher->msg_.header.stamp = ros::Time::now();
            publisher->unlockAndPublish();
        }
    }

private:

    // Joints handles.
    std::vector<hardware_interface::JointHandle> *joint_handles_ptr;

    // Realtime effort command publisher.
    typedef realtime_tools::RealtimePublisher<segmented_controller::Efforts> EffortsPublisher;
    boost::scoped_ptr<EffortsPublisher> publisher;

    // // Inverse Dynamics Solver.
    // boost::scoped_ptr<KDL::ChainIdSolver_RNE> id_solver;

    // Joints state.
    KDL::JntArrayVel joints_state;

    // Joints commands.
    KDL::JntArray
            joints_effort_limits,
            inner_loop_control,
            outer_loop_control;

    // Joints efforts.
    std::vector<double> *joints_efforts;

    std::vector<double> joint_positions;
    std::vector<double> joint_velocities;
    std::vector<double> joint_efforts;
};