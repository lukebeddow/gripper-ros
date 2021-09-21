#pragma once

#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/PhysicsTypes.hh>

#include <armadillo>
#include <std_msgs/Float64.h>

namespace gazebo
{
  class FingerPlugin : public ModelPlugin
  {
  public:

    /* member functions */
    FingerPlugin();
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
    void CreatePID(std::string joint_name);
    void EnforceLimit(std::string joint_name);
    void HardLimit(std::string joint_name);
    void testLimit(std::string joint_name);
    void update(const common::UpdateInfo &_info);

    arma::vec getJointValues(int finger_num);
    arma::mat getFingerXY(int finger_num);
    arma::vec getFingerCurve(int finger_num, int order);
    double getGaugeReading(int finger_num);
    void calculateGaugeReadings();
    void publishGaugeReadings();

    /* member variables */
    physics::ModelPtr model;
    event::ConnectionPtr updateConnection;
    std::vector<std::string> joint_names;
    int num_segments;
    int num_joints;
    double segment_length;
    int fit_order = 3;
    double gauge_x = 50 * 0.001;
    double update_rate = 10;
    common::Time wait_time_ns;
    common::Time last_update_ns { 0.0 };

    ros::NodeHandle nh;
    ros::Publisher pub_g1;
    ros::Publisher pub_g2;
    ros::Publisher pub_g3;
    std_msgs::Float64 msg_g1;
    std_msgs::Float64 msg_g2;
    std_msgs::Float64 msg_g3;
    
  };
}