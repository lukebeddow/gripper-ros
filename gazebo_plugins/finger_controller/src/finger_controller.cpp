#include <finger_controller.h>

namespace gazebo
{

  FingerPlugin::FingerPlugin() : ModelPlugin()
  {
    /* constructor */
  }

  void FingerPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    /* initialiser */

    model = _model;

    nh = ros::NodeHandle("FingerPlugin");
    pub_g1 = nh.advertise<std_msgs::Float64>("/Finger/gauge1", 10);
    pub_g2 = nh.advertise<std_msgs::Float64>("/Finger/gauge2", 10);
    pub_g3 = nh.advertise<std_msgs::Float64>("/Finger/gauge3", 10);

    // get the publishing rate
    if (!_sdf->HasElement("sdf_parameter_update_rate")) {
      ROS_WARN("update rate not set for finger plugin, default to 10Hz");
      update_rate = 10;
    }
    else {
      update_rate = _sdf->GetElement("sdf_parameter_update_rate")->Get<double>();
    }

    // wait time is 0.0 secs, 1/r nsecs
    wait_time_ns = common::Time(0.0, common::Time::SecToNano(1.0 / update_rate));
    last_update_ns = common::Time::GetWallTime();

    // get the number of segments
    if (!nh.getParam("/gripper_config/num_segments", num_segments)) {
      ROS_ERROR("could not find the parameter 'num_segments'");
      throw std::runtime_error("could not find the parameter 'num_segments'");
    }
    
    // calculate the length of each segment
    segment_length = 0.001 * (190 / num_segments);

    num_joints = num_segments - 1;

    // bind our callback function to the gazebo update event
    updateConnection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&FingerPlugin::update, this, std::placeholders::_1));

    // create a list of every joint
    for (int i = 0; i < 3; i++) {
      std::string finger = "finger_" + std::to_string(i + 1);
      for (int j = 0; j < num_joints; j++) {
        std::string joint = "_segment_joint_" + std::to_string(j + 1);
        joint_names.push_back(finger + joint);
      }
    }
  }

  void FingerPlugin::CreatePID(std::string joint_name)
  {
    /* Create a PID controller for a given joint */

    double kp = 100;
    double kd = 0;
    double ki = 0;

    this->model->GetJointController().reset(new physics::JointController(model));
    this->model->GetJointController()->AddJoint(model->GetJoint(joint_name));
    std::string name = model->GetJoint(joint_name)->GetScopedName();
    this->model->GetJointController()->SetVelocityPID(name, common::PID(kp, kd, ki));
  }

  void FingerPlugin::EnforceLimit(std::string joint_name)
  {
    /* This function checks if velocity is outside limits and if so it overrides it */

    physics::JointPtr joint = this->model->GetJoint(joint_name);
    double velocity = joint->GetVelocity(0);

    double v_lim = 0.1;

    if (velocity > v_lim) {
      CreatePID(joint_name);
      this->model->GetJointController()->SetVelocityTarget(joint->GetScopedName(), v_lim);
    }
    else if (velocity < v_lim * -1) {
      CreatePID(joint_name);
      this->model->GetJointController()->SetVelocityTarget(joint->GetScopedName(), v_lim * -1);
    }
  }

  void FingerPlugin::HardLimit(std::string joint_name)
  {
    /* Enforce a hard limit on velocity */

    physics::JointPtr joint = this->model->GetJoint(joint_name);
    double velocity = joint->GetVelocity(0);

    double v_lim = 0.5;

    // FOR TESTING
    // joint->SetPosition(0, 0.5, true);
    

    if (velocity > v_lim) {
      joint->SetVelocity(0, 0);
    }
    else if (velocity < v_lim * -1) {
      joint->SetVelocity(0, 0);
    }
    else {
      return;
    }

    gzmsg << joint_name << " had velocity changed from " << velocity << " to 0\n";
    joint->Update();
  }

  void FingerPlugin::testLimit(std::string joint_name)
  {
    /* testing */

    physics::JointPtr joint = this->model->GetJoint(joint_name);
    joint->SetVelocityLimit(0, 1.0);
    joint->SetEffortLimit(0, 1.0);

    physics::JointWrench jw = joint->GetForceTorque(0);

    ignition::math::Vector3d r = jw.body1Torque;

    //     ROS_INFO_STREAM(joint_name << ":\n"
    // << "b1F: " << jw.body1Force[0] << ", " << jw.body1Force[1] << ", " << jw.body1Force[2] << '\n'
    // << "b1T: " << jw.body1Torque[0] << ", " << jw.body1Torque[1] << ", " << jw.body1Torque[2] << '\n'
    // << "b2F: " << jw.body2Force[0] << ", " << jw.body2Force[1] << ", " << jw.body2Force[2] << '\n'
    // << "b2T: " << jw.body1Torque[0] << ", " << jw.body1Torque[1] << ", " << jw.body1Torque[2] << '\n'
    //     );

    // ROS_INFO_STREAM("Resultant for " << joint_name << ": x = " << r[0] << ", y = " 
    //   << r[1] << ", z = " << r[2]);

    // joint->CheckAndTruncateForce();
  }

  arma::vec FingerPlugin::getJointValues(int finger_num)
  {
    /* This function gets the value of every finger joint and saves them */

    if (finger_num != 1 and
        finger_num != 2 and
        finger_num != 3) {
        throw std::runtime_error("finger_num must be 1, 2, or 3");
    }

    arma::vec joint_vector(num_joints, arma::fill::zeros);
    int i = (finger_num - 1) * num_joints;

    for (int j = 0; j < num_joints; j++) {
      // extract the joint angle in radians
      joint_vector(j) = this->model->GetJoint(joint_names[i + j])->Position(0);
    }

    // ROS_INFO_STREAM("joint_vector is:\n " << joint_vector);

    return joint_vector;
  }

  arma::mat FingerPlugin::getFingerXY(int finger_num)
  {
    /* This function saves the X and Y coordinates of a finger */

    // retrieve the joint values for this finger
    arma::vec joint_angles = getJointValues(finger_num);

    // initialise vector and matrix
    arma::vec cumulative(num_joints, arma::fill::zeros);
    arma::mat finger_xy(num_segments, 2, arma::fill::zeros);

    // calculate the x and y positions of the finger joints, end and tip inclusive
    for (int i = 0; i < num_joints; i++) {
      // keep cumulative total of angular sum
      if (i == 0) {
        cumulative(0) = joint_angles(0);
      }
      else {
        cumulative(i) = cumulative(i - 1) + joint_angles(i);
      }
      // tally up the cartesian coordinates of the segment tips
      finger_xy(i + 1, 0) = finger_xy(i, 0) + segment_length * std::cos(cumulative[i]);
      finger_xy(i + 1, 1) = finger_xy(i, 1) + segment_length * std::sin(cumulative[i]);
    }

    // ROS_INFO_STREAM("cumulative angles are:\n" << cumulative);
    // ROS_INFO_STREAM("finger_xy is:\n " << finger_xy);

    return finger_xy;
  }

  arma::vec FingerPlugin::getFingerCurve(int finger_num, int order)
  {
    /* This function gets the fit parameters for a curve fit of specified order */

    // retrieve the x and y positions of each finger joint (tip inclusive)
    arma::mat finger_xy = getFingerXY(finger_num);

    // polyfit the points to a curve
    arma::vec coeff = arma::polyfit(finger_xy.col(0), finger_xy.col(1), order);

    // ROS_INFO_STREAM("fit coefficients are:\n " << coeff);

    return coeff;
  }

  double FingerPlugin::getGaugeReading(int finger_num)
  {
    /* This function approximates the curve of a finger, and then uses this to
    get deflection, and hence strain, to calculate a gauge reading */

    // get the curve fit coefficients for this finger
    arma::vec coeff = getFingerCurve(finger_num, fit_order);

    // evaluate y at the gauge position
    double y = 0.0;
    for (int i = 0; i <= fit_order; i++) {
      y += coeff(i) * std::pow(gauge_x, fit_order - i);
    }

    /* delta = (P * l^3) / (3 * E * I), hence:
       delta = k * l^3, therefore:
           k = delta / l^3
    Since k is proportional to force, we assume it is proportional to strain as
    well. Getting SI units is not necessary, using an empirical calibration is
    easier */
    double k = y / std::cbrt(gauge_x);

    // ROS_INFO_STREAM("Delta is: " << y << ", the gauge reading is: " << k);

    return k;
  }

  void FingerPlugin::calculateGaugeReadings()
  {
    /* This function calculates the strain gauge reading for each of the three
    fingers based on an approximation of the deflection */

    // displacement should be cubic wrt length
    fit_order = 2;

    // get the gauge readings
    double k1 = getGaugeReading(1);
    double k2 = getGaugeReading(2);
    double k3 = getGaugeReading(3);

    // scale and offset factors
    double scale_1 = 1e6;
    double scale_2 = 1e6;
    double scale_3 = 1e6;
    double offset_1 = 0.0;
    double offset_2 = 0.0;
    double offset_3 = 0.0;

    // apply scalings
    k1 = k1 * scale_1 + offset_1;
    k2 = k2 * scale_2 + offset_2;
    k3 = k3 * scale_3 + offset_3;

    // save
    msg_g1.data = k1;
    msg_g2.data = k2;
    msg_g3.data = k3;
    
  }

  void FingerPlugin::publishGaugeReadings()
  {
    /* This function calculates and publishes strain gauge readings */

    // calculate
    calculateGaugeReadings();

    // publish
    pub_g1.publish(msg_g1);
    pub_g2.publish(msg_g2);
    pub_g3.publish(msg_g3);
  }

  void FingerPlugin::update(const common::UpdateInfo &_info)
  {

    static const int update_limit = 1000;
    static int update_num = 0;

    // if enough time has elapsed since our last publish
    common::Time time_now_ns = common::Time::GetWallTime();
    if (time_now_ns - last_update_ns > wait_time_ns) {
      publishGaugeReadings();
      last_update_ns = time_now_ns;
    }

    // if (update_num < update_limit) {
        // ROS_INFO_STREAM("Gauge update number " << update_num);
        // publishGaugeReadings();
        // update_num++;
    // }
    
    
    // this->model->GetJointController()->Reset();

    // for (int i = 0; i < joint_names.size(); i++) {
    //   testLimit(joint_names[i]);
    // }

    // this->model->GetJointController()->Update();
    

  }

  GZ_REGISTER_MODEL_PLUGIN(FingerPlugin)
}