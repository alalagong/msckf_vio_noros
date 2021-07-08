/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include "msckf_vio/params.h"
#include "msckf_vio.h"
#include "math_utils.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>

#ifdef USING_SPARSE_QR
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#endif

#include "opencv2/opencv.hpp"

namespace msckf_vio {

MsckfVio::MsckfVio() :
    is_gravity_set(false),
    is_first_img(true)
{

}

bool MsckfVio::initialize(
    Stereo_camera_config_t& stereo_param)
{
    if (!loadParameters(stereo_param)) {
        return false;
    }
    std::cout << "Finish loading parameters..." << std::endl;

    // Initialize state server
    state_server.continuous_noise_cov =
        Eigen::Matrix<double, 12, 12>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) =
        Eigen::Matrix3d::Identity() * IMUState::gyro_noise;
    state_server.continuous_noise_cov.block<3, 3>(3, 3) =
        Eigen::Matrix3d::Identity() * IMUState::gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(6, 6) =
        Eigen::Matrix3d::Identity() * IMUState::acc_noise;
    state_server.continuous_noise_cov.block<3, 3>(9, 9) =
        Eigen::Matrix3d::Identity() * IMUState::acc_bias_noise;

    // Initialize the chi squared test table with confidence
    // level 0.95.
    for (uint32_t i = 1; i < 100; ++i) {
        chi_squared_test_table[i] = chi2inv(0.95, i);
    }
    return true;
}

bool MsckfVio::loadParameters(Stereo_camera_config_t& params)
{
    // Frame id
    fixed_frame_id = param_fixed_frame_id;
    child_frame_id = param_child_frame_id;
    frame_rate = param_frame_rate;
    position_std_threshold = param_position_std_threshold;

    rotation_threshold = param_rotation_threshold;
    translation_threshold = param_translation_threshold;
    tracking_rate_threshold = param_tracking_rate_threshold;

    // Feature optimization parameters
    Feature::optimization_config.translation_threshold =
        param_feature_translation_threshold;

    // Noise related parameters
    IMUState::gyro_noise = param_noise_gyro;
    IMUState::acc_noise = param_noise_acc;
    IMUState::gyro_bias_noise = param_noise_gyro_bias;
    IMUState::acc_bias_noise = param_noise_acc_bias;
    Feature::observation_noise = param_noise_feature;

    // Use variance instead of standard deviation.
    IMUState::gyro_noise *= IMUState::gyro_noise;
    IMUState::acc_noise *= IMUState::acc_noise;
    IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
    IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
    Feature::observation_noise *= Feature::observation_noise;

    // Set the initial IMU state.
    // The intial orientation and position will be set to the origin
    // implicitly. But the initial velocity and bias can be
    // set by parameters.
    // TODO: is it reasonable to set the initial bias to 0?
    state_server.imu_state.velocity(0) = param_initial_state_velocity_x;
    state_server.imu_state.velocity(1) = param_initial_state_velocity_y;
    state_server.imu_state.velocity(2) = param_initial_state_velocity_z;

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    state_server.state_cov = Eigen::MatrixXd::Zero(21, 21);
    for (int32_t i = 3; i < 6; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_gryo_bias;
    }
    for (int32_t i = 6; i < 9; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_velocity;
    }
    for (int32_t i = 9; i < 12; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_acc_bias;
    }
    for (int32_t i = 15; i < 18; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_rot;
    }
    for (int32_t i = 18; i < 21; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_trans;
    }

    // Transformation offsets between the frames involved.
    Eigen::Isometry3d T_cam0_imu(Eigen::Isometry3d::Identity());
    T_cam0_imu.translation() = Eigen::Vector3d(params.t_cam0_imu);
    T_cam0_imu.linear() =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(params.R_cam0_imu);

    Eigen::Isometry3d T_cam1_imu(Eigen::Isometry3d::Identity());
    T_cam1_imu.translation() = Eigen::Vector3d(params.t_cam1_imu);
    T_cam1_imu.linear() =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(params.R_cam1_imu);

    state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
    state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();

    CAMState::T_cam0_cam1 = T_cam1_imu.inverse() * T_cam0_imu;

    // Maximum number of camera states to be stored
    max_cam_state_size = param_max_cam_state_size;

    return true;
}

void MsckfVio::reset()
{
    std::cout << "Start resetting msckf vio..." << std::endl;

    // Reset the IMU state.
    IMUState& imu_state = state_server.imu_state;
    imu_state.time = 0;
    imu_state.orientation = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    imu_state.position = Eigen::Vector3d::Zero();
    imu_state.velocity = Eigen::Vector3d::Zero();
    imu_state.gyro_bias = Eigen::Vector3d::Zero();
    imu_state.acc_bias = Eigen::Vector3d::Zero();
    imu_state.orientation_null = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    imu_state.position_null = Eigen::Vector3d::Zero();
    imu_state.velocity_null = Eigen::Vector3d::Zero();

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Reset the state covariance.
    state_server.imu_state.velocity(0) = param_initial_state_velocity_x;
    state_server.imu_state.velocity(1) = param_initial_state_velocity_y;
    state_server.imu_state.velocity(2) = param_initial_state_velocity_z;

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    state_server.state_cov = Eigen::MatrixXd::Zero(21, 21);
    for (int32_t i = 3; i < 6; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_gryo_bias;
    }
    for (int32_t i = 6; i < 9; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_velocity;
    }
    for (int32_t i = 9; i < 12; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_acc_bias;
    }
    for (int32_t i = 15; i < 18; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_rot;
    }
    for (int32_t i = 18; i < 21; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_trans;
    }

    // Clear all exsiting features in the map.
    map_server.clear();

    // Reset the starting flags.
    is_gravity_set = false;
    is_first_img = true;

    std::cout << "Resetting msckf vio completed..." << std::endl;
}

void MsckfVio::initializeGravityAndBias(
    const std::deque<Sensor_imu_t> &imu_buffer)
{
    // Initialize gravity and gyro bias.
    Eigen::Vector3d sum_angular_vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d sum_linear_acc = Eigen::Vector3d::Zero();

    for (const auto& imu : imu_buffer) {
        Eigen::Vector3d angular_vel(
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z);
        Eigen::Vector3d linear_acc(
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z);

        sum_angular_vel += angular_vel;
        sum_linear_acc += linear_acc;
    }

    state_server.imu_state.gyro_bias =
        sum_angular_vel / imu_buffer.size();
    //IMUState::gravity =
    //  -sum_linear_acc / imu_buffer.size();
    // This is the gravity in the IMU frame.
    Eigen::Vector3d gravity_imu =
        sum_linear_acc / imu_buffer.size();

    // Initialize the initial orientation, so that the estimation
    // is consistent with the inertial frame.
    double gravity_norm = gravity_imu.norm();
    IMUState::gravity = Eigen::Vector3d(0.0, 0.0, -gravity_norm);

    Eigen::Quaterniond q0_i_w = Eigen::Quaterniond::FromTwoVectors(
        gravity_imu, -IMUState::gravity);
    state_server.imu_state.orientation =
        rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

    is_gravity_set = true;
}

void MsckfVio::featureCallback(
    const Feature_measure_t& measure,
    const std::deque<Sensor_imu_t>& imu_buffer)
{
    if (!is_gravity_set) {
        return;
    }

    // Start the system if the first image is received.
    // The frame where the first image is received will be
    // the origin.
    if (is_first_img) {
        is_first_img = false;
        state_server.imu_state.time = measure.stamp;
    }

    // Propogate the IMU state.
    // that are received before the image msg.
    batchImuProcessing(measure.stamp, imu_buffer);

    // Augment the state vector.
    stateAugmentation(measure.stamp);

    // Add new observations for existing features or new
    // features in the map server.
    addFeatureObservations(measure);

    // Perform measurement update if necessary.
    removeLostFeatures();

    pruneCamStateBuffer();

    // Reset the system if necessary.
    onlineReset();
}

void MsckfVio::batchImuProcessing(
    const uint64_t& time_bound,
    const std::deque<Sensor_imu_t> &imu_buffer)
{
    uint32_t imu_buffer_size = (uint32_t)imu_buffer.size();
    for (uint32_t i = 0; i < imu_buffer_size; i++) {
        const Sensor_imu_t& imu = imu_buffer[i];
        const uint64_t& imu_time = imu.stamp;
        if (imu_time <= state_server.imu_state.time) {
            continue;
        }
        if (imu_time > time_bound) {
            break;
        }

        // Convert the msgs.
        Eigen::Vector3d m_gyro(
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z);
        Eigen::Vector3d m_acc(
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z);

        // Execute process model.
        processModel(imu_time, m_gyro, m_acc);
    }

    // Set the state ID for the new IMU state.
    state_server.imu_state.id = IMUState::next_id++;
}

void MsckfVio::processModel(
    const uint64_t& time,
    const Eigen::Vector3d& m_gyro,
    const Eigen::Vector3d& m_acc)
{
    // Remove the bias from the measured gyro and acceleration
    IMUState& imu_state = state_server.imu_state;
    Eigen::Vector3d gyro = m_gyro - imu_state.gyro_bias;
    Eigen::Vector3d acc = m_acc - imu_state.acc_bias;
    double dtime = ((double)time - (double)imu_state.time) / 1e9;

    // Compute discrete transition and noise covariance matrix
    Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
    Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();

    F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -quaternionToRotation(
        imu_state.orientation).transpose() * skewSymmetric(acc);
    F.block<3, 3>(6, 9) = -quaternionToRotation(
        imu_state.orientation).transpose();
    F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

    G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -quaternionToRotation(
        imu_state.orientation).transpose();
    G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s.
    Eigen::Matrix<double, 21, 21> Fdt = F * dtime;
    Eigen::Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Eigen::Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
    Eigen::Matrix<double, 21, 21> Phi =
        Eigen::Matrix<double, 21, 21>::Identity() +
        Fdt + 0.5 * Fdt_square + (1.0 / 6.0) * Fdt_cube;

    // Propogate the state using 4th order Runge-Kutta
    predictNewState(dtime, gyro, acc);

    // Modify the transition matrix
    Eigen::Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
    Phi.block<3, 3>(0, 0) =
        quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

    Eigen::Vector3d u = R_kk_1 * IMUState::gravity;
    Eigen::RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

    Eigen::Matrix3d A1 = Phi.block<3, 3>(6, 0);
    Eigen::Vector3d w1 = skewSymmetric(
        imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
    Phi.block<3, 3>(6, 0) = A1 - (A1 * u - w1) * s;

    Eigen::Matrix3d A2 = Phi.block<3, 3>(12, 0);
    Eigen::Vector3d w2 = skewSymmetric(
        dtime * imu_state.velocity_null + imu_state.position_null-
        imu_state.position) * IMUState::gravity;
    Phi.block<3, 3>(12, 0) = A2 - (A2 * u - w2) * s;

    // Propogate the state covariance matrix.
    Eigen::Matrix<double, 21, 21> Q =
        Phi * G * state_server.continuous_noise_cov
        * G.transpose() * Phi.transpose() * dtime;
    state_server.state_cov.block<21, 21>(0, 0) =
        Phi * state_server.state_cov.block<21, 21>(0, 0)
        * Phi.transpose() + Q;

    if (state_server.cam_states.size() > 0) {
        state_server.state_cov.block(
            0, 21, 21, state_server.state_cov.cols() - 21) =
            Phi * state_server.state_cov.block(
            0, 21, 21, state_server.state_cov.cols() - 21);
        state_server.state_cov.block(
            21, 0, state_server.state_cov.rows() - 21, 21) =
            state_server.state_cov.block(
            21, 0, state_server.state_cov.rows() - 21, 21) * Phi.transpose();
    }

    Eigen::MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state correspondes to null space.
    imu_state.orientation_null = imu_state.orientation;
    imu_state.position_null = imu_state.position;
    imu_state.velocity_null = imu_state.velocity;

    // Update the state info
    state_server.imu_state.time = time;
}

void MsckfVio::predictNewState(
    const double& dt,
    const Eigen::Vector3d& gyro,
    const Eigen::Vector3d& acc)
{
    // TODO: Will performing the forward integration using
    //    the inverse of the quaternion give better accuracy?
    double gyro_norm = gyro.norm();
    Eigen::Matrix4d Omega = Eigen::Matrix4d::Zero();
    Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
    Omega.block<3, 1>(0, 3) = gyro;
    Omega.block<1, 3>(3, 0) = -gyro;

    Eigen::Vector4d& q = state_server.imu_state.orientation;
    Eigen::Vector3d& v = state_server.imu_state.velocity;
    Eigen::Vector3d& p = state_server.imu_state.position;

    // Some pre-calculation
    Eigen::Vector4d dq_dt, dq_dt2;
    if (gyro_norm > 1e-5) {
        dq_dt = (cos(gyro_norm * dt * 0.5) * Eigen::Matrix4d::Identity() +
            1 / gyro_norm * sin(gyro_norm * dt * 0.5) * Omega) * q;
        dq_dt2 = (cos(gyro_norm * dt * 0.25) * Eigen::Matrix4d::Identity() +
            1 / gyro_norm * sin(gyro_norm * dt * 0.25) * Omega) * q;
    }
    else {
        dq_dt = (Eigen::Matrix4d::Identity() + 0.5 * dt * Omega) *
            cos(gyro_norm * dt * 0.5) * q;
        dq_dt2 = (Eigen::Matrix4d::Identity() + 0.25 * dt * Omega) *
            cos(gyro_norm * dt * 0.25) * q;
    }
    Eigen::Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
    Eigen::Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

    // k1 = f(tn, yn)
    Eigen::Vector3d k1_v_dot =
        quaternionToRotation(q).transpose() * acc + IMUState::gravity;
    Eigen::Vector3d k1_p_dot = v;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    Eigen::Vector3d k1_v = v + k1_v_dot * dt / 2;
    Eigen::Vector3d k2_v_dot =
        dR_dt2_transpose * acc + IMUState::gravity;
    Eigen::Vector3d k2_p_dot = k1_v;

    // k3 = f(tn+dt/2, yn+k2*dt/2)
    Eigen::Vector3d k2_v = v + k2_v_dot * dt / 2;
    Eigen::Vector3d k3_v_dot =
        dR_dt2_transpose * acc + IMUState::gravity;
    Eigen::Vector3d k3_p_dot = k2_v;

    // k4 = f(tn+dt, yn+k3*dt)
    Eigen::Vector3d k3_v = v + k3_v_dot * dt;
    Eigen::Vector3d k4_v_dot =
        dR_dt_transpose * acc + IMUState::gravity;
    Eigen::Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    q = dq_dt;
    quaternionNormalize(q);
    v = v + dt / 6 * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
    p = p + dt / 6 * (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}

void MsckfVio::stateAugmentation(const uint64_t& time)
{
    const Eigen::Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
    const Eigen::Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

    // Add a new camera state to the state server.
    Eigen::Matrix3d R_w_i = quaternionToRotation(
        state_server.imu_state.orientation);
    Eigen::Matrix3d R_w_c = R_i_c * R_w_i;
    Eigen::Vector3d t_c_w = state_server.imu_state.position +
        R_w_i.transpose() * t_c_i;

    state_server.cam_states[state_server.imu_state.id] =
        CAMState(state_server.imu_state.id);
    CAMState& cam_state = state_server.cam_states[state_server.imu_state.id];

    cam_state.time = time;
    cam_state.orientation = rotationToQuaternion(R_w_c);
    cam_state.position = t_c_w;

    cam_state.orientation_null = cam_state.orientation;
    cam_state.position_null = cam_state.position;

    // Update the covariance matrix of the state.
    // To simplify computation, the matrix J below is the nontrivial block
    // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
    // -aided Inertial Navigation".
    Eigen::Matrix<double, 6, 21> J = Eigen::Matrix<double, 6, 21>::Zero();
    J.block<3, 3>(0, 0) = R_i_c;
    J.block<3, 3>(0, 15) = Eigen::Matrix3d::Identity();
    J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
    //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
    J.block<3, 3>(3, 12) = Eigen::Matrix3d::Identity();
    J.block<3, 3>(3, 18) = Eigen::Matrix3d::Identity();

    // Resize the state covariance matrix.
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

    // Rename some matrix blocks for convenience.
    const Eigen::Matrix<double, 21, 21>& P11 =
        state_server.state_cov.block<21, 21>(0, 0);
    const Eigen::MatrixXd& P12 =
        state_server.state_cov.block(0, 21, 21, old_cols-21);

    // Fill in the augmented state covariance.
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;
    state_server.state_cov.block(0, old_cols, old_rows, 6) =
        state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
    state_server.state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    Eigen::MatrixXd state_cov_fixed = (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
}

void MsckfVio::addFeatureObservations(
    const Feature_measure_t& msg)
{
    StateIDType state_id = state_server.imu_state.id;
    uint32_t curr_feature_num = (uint32_t)map_server.size();
    uint32_t tracked_feature_num = 0;

    // Add new observations for existing features or new
    // features in the map server.
    for (const auto& feature : msg.features) {
        if (map_server.find(feature.id) == map_server.end()) {
            // This is a new feature.
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] =
                Eigen::Vector4d(feature.u0, feature.v0,
                    feature.u1, feature.v1);
        } else {
            // This is an old feature.
            map_server[feature.id].observations[state_id] =
            Eigen::Vector4d(feature.u0, feature.v0,
                feature.u1, feature.v1);
            ++tracked_feature_num;
        }
    }

    tracking_rate =
        static_cast<double>(tracked_feature_num) /
        static_cast<double>(curr_feature_num);
}

void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Eigen::Matrix<double, 4, 6>& H_x,
    Eigen::Matrix<double, 4, 3>& H_f,
    Eigen::Vector4d& r)
{
    // Prepare all the required data.
    const CAMState& cam_state = state_server.cam_states[cam_state_id];
    const Feature& feature = map_server[feature_id];

    // Cam0 pose.
    Eigen::Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
    const Eigen::Vector3d& t_c0_w = cam_state.position;

    // Cam1 pose.
    Eigen::Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
    Eigen::Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
    Eigen::Vector3d t_c1_w =
        t_c0_w - R_w_c1.transpose() * CAMState::T_cam0_cam1.translation();

    // 3d feature position in the world frame.
    // And its observation with the stereo cameras.
    const Eigen::Vector3d& p_w = feature.position;
    const Eigen::Vector4d& z =
        feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to
    // the cam0 and cam1 frame.
    Eigen::Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
    Eigen::Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

    // Compute the Jacobians.
    Eigen::Matrix<double, 4, 3> dz_dpc0 =
        Eigen::Matrix<double, 4, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

    Eigen::Matrix<double, 4, 3> dz_dpc1 =
        Eigen::Matrix<double, 4, 3>::Zero();
    dz_dpc1(2, 0) = 1 / p_c1(2);
    dz_dpc1(3, 1) = 1 / p_c1(2);
    dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
    dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

    Eigen::Matrix<double, 3, 6> dpc0_dxc =
        Eigen::Matrix<double, 3, 6>::Zero();
    dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
    dpc0_dxc.rightCols(3) = -R_w_c0;

    Eigen::Matrix<double, 3, 6> dpc1_dxc =
        Eigen::Matrix<double, 3, 6>::Zero();
    dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
    dpc1_dxc.rightCols(3) = -R_w_c1;

    Eigen::Matrix3d dpc0_dpg = R_w_c0;
    Eigen::Matrix3d dpc1_dpg = R_w_c1;

    H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
    H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

    // Modifty the measurement Jacobian to ensure
    // observability constrain.
    Eigen::Matrix<double, 4, 6> A = H_x;
    Eigen::Matrix<double, 6, 1> u =
        Eigen::Matrix<double, 6, 1>::Zero();
    u.block<3, 1>(0, 0) =
        quaternionToRotation(cam_state.orientation_null) *
        IMUState::gravity;
    u.block<3, 1>(3, 0) =
        skewSymmetric(p_w - cam_state.position_null) *
        IMUState::gravity;
    H_x = A - A * u * (u.transpose() * u).inverse() * u.transpose();
    H_f = -H_x.block<4, 3>(0, 3);

    // Compute the residual.
    r = z - Eigen::Vector4d(
        p_c0(0) / p_c0(2),
        p_c0(1) / p_c0(2),
        p_c1(0) / p_c1(2),
        p_c1(1) / p_c1(2));
}

void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    Eigen::MatrixXd& H_x,
    Eigen::VectorXd& r)
{
    const auto& feature = map_server[feature_id];

    // Check how many camera states in the provided camera
    // id camera has actually seen this feature.
    std::vector<StateIDType> valid_cam_state_ids(0);
    for (const auto& cam_id : cam_state_ids) {
        if (feature.observations.find(cam_id) ==
            feature.observations.end()) {
                continue;
        }

        valid_cam_state_ids.push_back(cam_id);
    }

    uint32_t jacobian_row_size = 4 * (uint32_t)valid_cam_state_ids.size();

    Eigen::MatrixXd H_xj = Eigen::MatrixXd::Zero(jacobian_row_size,
        21 + state_server.cam_states.size() * 6);
    Eigen::MatrixXd H_fj = Eigen::MatrixXd::Zero(jacobian_row_size, 3);
    Eigen::VectorXd r_j = Eigen::VectorXd::Zero(jacobian_row_size);
    int32_t stack_cntr = 0;

    for (const auto& cam_id : valid_cam_state_ids) {
        Eigen::Matrix<double, 4, 6> H_xi =
            Eigen::Matrix<double, 4, 6>::Zero();
        Eigen::Matrix<double, 4, 3> H_fi =
            Eigen::Matrix<double, 4, 3>::Zero();
        Eigen::Vector4d r_i = Eigen::Vector4d::Zero();
        measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

        auto cam_state_iter = state_server.cam_states.find(cam_id);
        uint32_t cam_state_cntr = (uint32_t)std::distance(
            state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace of H_fj.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(
        H_fj, Eigen::ComputeFullU | Eigen::ComputeThinV);
    Eigen::MatrixXd A =
        svd_helper.matrixU().rightCols(jacobian_row_size - 3);

    H_x = A.transpose() * H_xj;
    r = A.transpose() * r_j;
}

void MsckfVio::measurementUpdate(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& r)
{
    if (H.rows() == 0 || r.rows() == 0) {
        return;
    }

    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    Eigen::MatrixXd H_thin;
    Eigen::VectorXd r_thin;

    if (H.rows() > H.cols()) {
#ifdef USING_SPARSE_QR
        // Convert H to a sparse matrix.
        Eigen::SparseMatrix<double> H_sparse = H.sparseView();

        // Perform QR decompostion on H_sparse.
        Eigen::SPQR<Eigen::SparseMatrix<double> > spqr_helper;
        spqr_helper.compute(H_sparse);

        Eigen::MatrixXd H_temp;
        Eigen::VectorXd r_temp;
        (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
        (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

        H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
        r_thin = r_temp.head(21+state_server.cam_states.size()*6);
#else
        Eigen::HouseholderQR<Eigen::MatrixXd> qr_helper(H);
        Eigen::MatrixXd Q = qr_helper.householderQ();
        Eigen::MatrixXd Q1 = Q.leftCols(21 + state_server.cam_states.size() * 6);

        H_thin = Q1.transpose() * H;
        r_thin = Q1.transpose() * r;
#endif
    }
    else {
        H_thin = H;
        r_thin = r;
    }

    // Compute the Kalman gain.
    const Eigen::MatrixXd& P = state_server.state_cov;
    Eigen::MatrixXd S = H_thin * P * H_thin.transpose() +
        Feature::observation_noise * Eigen::MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
    //Eigen::MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
    Eigen::MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
    Eigen::MatrixXd K = K_transpose.transpose();

    // Compute the error of the state.
    Eigen::VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const Eigen::VectorXd& delta_x_imu = delta_x.head<21>();

    if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
        //delta_x_imu.segment<3>(3).norm() > 0.15 ||
        delta_x_imu.segment<3>(6).norm() > 0.5 ||
        //delta_x_imu.segment<3>(9).norm() > 0.5 ||
        delta_x_imu.segment<3>(12).norm() > 1.0) {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
        printf("Update change is too large.\n");
        //return;
    }

    const Eigen::Vector4d dq_imu =
        smallAngleQuaternion(delta_x_imu.head<3>());
    state_server.imu_state.orientation = quaternionMultiplication(
        dq_imu, state_server.imu_state.orientation);
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
    state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.position += delta_x_imu.segment<3>(12);

    const Eigen::Vector4d dq_extrinsic =
        smallAngleQuaternion(delta_x_imu.segment<3>(15));
    state_server.imu_state.R_imu_cam0 =
        quaternionToRotation(dq_extrinsic) *
        state_server.imu_state.R_imu_cam0;
    state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (uint32_t i = 0; i < state_server.cam_states.size();
        ++i, ++cam_state_iter) {
        const Eigen::VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
        const Eigen::Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
        cam_state_iter->second.orientation = quaternionMultiplication(
            dq_cam, cam_state_iter->second.orientation);
        cam_state_iter->second.position += delta_x_cam.tail<3>();
    }

    // Update state covariance.
    Eigen::MatrixXd I_KH =
        Eigen::MatrixXd::Identity(K.rows(), H_thin.cols()) - K * H_thin;
    //state_server.state_cov = 
    //    I_KH * state_server.state_cov * I_KH.transpose() +
    //    K * K.transpose() * Feature::observation_noise;
    state_server.state_cov = I_KH * state_server.state_cov;

    // Fix the covariance to be symmetric
    Eigen::MatrixXd state_cov_fixed =
        (state_server.state_cov +
        state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;
}

bool MsckfVio::gatingTest(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& r,
    const uint32_t& dof)
{
    Eigen::MatrixXd P1 = H * state_server.state_cov * H.transpose();
    Eigen::MatrixXd P2 = Feature::observation_noise *
        Eigen::MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

    //std::cout << dof << " " << gamma << " " <<
    //    chi_squared_test_table[dof] << " ";

    if (gamma >= chi_squared_test_table[dof]) {
        //std::cout << "failed" << std::endl;
        return false;
    }
    //std::cout << "passed" << std::endl;
    return true;
}

void MsckfVio::removeLostFeatures()
{
    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    uint32_t jacobian_row_size = 0;
    std::vector<FeatureIDType> invalid_feature_ids(0);
    std::vector<FeatureIDType> processed_feature_ids(0);

    for (auto iter = map_server.begin();
        iter != map_server.end(); ++iter) {
        // Rename the feature to be checked.
        auto& feature = iter->second;

        // Pass the features that are still being tracked.
        if (feature.observations.find(state_server.imu_state.id)
            != feature.observations.end()) {
            continue;
        }
        if (feature.observations.size() < 3) {
            invalid_feature_ids.push_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it
        // has not been.
        if (!feature.is_initialized) {
            if (!feature.checkMotion(state_server.cam_states)) {
                invalid_feature_ids.push_back(feature.id);
                continue;
            } else {
                if (!feature.initializePosition(state_server.cam_states)) {
                    invalid_feature_ids.push_back(feature.id);
                    continue;
                }
            }
        }

        jacobian_row_size += 4 * (uint32_t)feature.observations.size() - 3;
        processed_feature_ids.push_back(feature.id);
    }

    //std::cout << "invalid/processed feature #: " <<
    //    invalid_feature_ids.size() << "/" <<
    //    processed_feature_ids.size() << std::endl;
    //std::cout << "jacobian row #: " << jacobian_row_size << std::endl;

    // Remove the features that do not have enough measurements.
    for (const auto& feature_id : invalid_feature_ids) {
        map_server.erase(feature_id);
    }

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.size() == 0) {
        return;
    }

    Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(
        jacobian_row_size,
        21 + 6 * state_server.cam_states.size());
    Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
    uint32_t stack_cntr = 0;

    // Process the features which lose track.
    for (const auto& feature_id : processed_feature_ids) {
        auto& feature = map_server[feature_id];

        std::vector<StateIDType> cam_state_ids(0);
        for (const auto& measurement : feature.observations) {
            cam_state_ids.push_back(measurement.first);
        }

        Eigen::MatrixXd H_xj;
        Eigen::VectorXd r_j;
        featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, (uint32_t)cam_state_ids.size() - 1)) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += (uint32_t)H_xj.rows();
        }

        // Put an upper bound on the row size of measurement Jacobian,
        // which helps guarantee the executation time.
        if (stack_cntr > 1500) {
            break;
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    measurementUpdate(H_x, r);

    // Remove all processed features from the map.
    for (const auto& feature_id : processed_feature_ids) {
        map_server.erase(feature_id);
    }
}

void MsckfVio::findRedundantCamStates(
    std::vector<StateIDType>& rm_cam_state_ids)
{
    // Move the iterator to the key position.
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int32_t i = 0; i < 4; ++i) {
        --key_cam_state_iter;
    }
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;
    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    const Eigen::Vector3d key_position =
        key_cam_state_iter->second.position;
    const Eigen::Matrix3d key_rotation = quaternionToRotation(
        key_cam_state_iter->second.orientation);

    // Mark the camera states to be removed based on the
    // motion between states.
    for (int32_t i = 0; i < 2; ++i) {
        const Eigen::Vector3d position =
            cam_state_iter->second.position;
        const Eigen::Matrix3d rotation = quaternionToRotation(
            cam_state_iter->second.orientation);

        double distance = (position - key_position).norm();
        double angle = Eigen::AngleAxisd(
            rotation * key_rotation.transpose()).angle();

        //if (angle < 0.1745 && distance < 0.2 && tracking_rate > 0.5) {
        if (angle < 0.2618 && distance < 0.4 && tracking_rate > 0.5) {
            rm_cam_state_ids.push_back(cam_state_iter->first);
            ++cam_state_iter;
        } else {
            rm_cam_state_ids.push_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());
}

void MsckfVio::pruneCamStateBuffer()
{
    if ((int32_t)state_server.cam_states.size() < max_cam_state_size) {
        return;
    }

    // Find two camera states to be removed.
    std::vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

    // Find the size of the Jacobian matrix.
    int32_t jacobian_row_size = 0;
    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        std::vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) !=
                feature.observations.end())
            involved_cam_state_ids.push_back(cam_id);
        }

        if (involved_cam_state_ids.size() == 0) {
            continue;
        }
        if (involved_cam_state_ids.size() == 1) {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }

        if (!feature.is_initialized) {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states)) {
            // If the feature cannot be initialized, just remove
            // the observations associated with the camera states
            // to be removed.
            for (const auto& cam_id : involved_cam_state_ids)
                feature.observations.erase(cam_id);
                continue;
            } else {
                if(!feature.initializePosition(state_server.cam_states)) {
                    for (const auto& cam_id : involved_cam_state_ids) {
                        feature.observations.erase(cam_id);
                    }
                    continue;
                }
            }
        }

        jacobian_row_size += 4 * (int32_t)involved_cam_state_ids.size() - 3;
    }

    //std::cout << "jacobian row #: " << jacobian_row_size << std::endl;

    // Compute the Jacobian and residual.
    Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(
        jacobian_row_size,
        21 + 6 * state_server.cam_states.size());
    Eigen::VectorXd r = Eigen::VectorXd::Zero(jacobian_row_size);
    int32_t stack_cntr = 0;

    for (auto& item : map_server) {
        auto& feature = item.second;
        // Check how many camera states to be removed are associated
        // with this feature.
        std::vector<StateIDType> involved_cam_state_ids(0);
        for (const auto& cam_id : rm_cam_state_ids) {
            if (feature.observations.find(cam_id) !=
                feature.observations.end()) {
                involved_cam_state_ids.push_back(cam_id);
            }
        }

        if (involved_cam_state_ids.size() == 0) {
            continue;
        }

        Eigen::MatrixXd H_xj;
        Eigen::VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, (uint32_t)involved_cam_state_ids.size())) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += (int32_t)H_xj.rows();
        }

        for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform measurement update.
    measurementUpdate(H_x, r);

    for (const auto& cam_id : rm_cam_state_ids) {
        int32_t cam_sequence = (int32_t)std::distance(
            state_server.cam_states.begin(),
            state_server.cam_states.find(cam_id));
        int32_t cam_state_start = 21 + 6 * cam_sequence;
        int32_t cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state
        // covariance matrix.
        if (cam_state_end < state_server.state_cov.rows()) {
            state_server.state_cov.block(
                cam_state_start, 0,
                state_server.state_cov.rows() - cam_state_end,
                state_server.state_cov.cols()) =
                state_server.state_cov.block(cam_state_end, 0,
                state_server.state_cov.rows() - cam_state_end,
                state_server.state_cov.cols());

            state_server.state_cov.block(
                0, cam_state_start,
                state_server.state_cov.rows(),
                state_server.state_cov.cols() - cam_state_end) =
                state_server.state_cov.block(0, cam_state_end,
                state_server.state_cov.rows(),
                state_server.state_cov.cols() - cam_state_end);

            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6,
                state_server.state_cov.cols() - 6);
        } else {
            state_server.state_cov.conservativeResize(
                state_server.state_cov.rows() - 6,
                state_server.state_cov.cols() - 6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }
}

void MsckfVio::onlineReset()
{
    // Never perform online reset if position std threshold
    // is non-positive.
    if (position_std_threshold <= 0) {
        return;
    }
    static long long int online_reset_counter = 0;

    // Check the uncertainty of positions to determine if
    // the system can be reset.
    double position_x_std = std::sqrt(state_server.state_cov(12, 12));
    double position_y_std = std::sqrt(state_server.state_cov(13, 13));
    double position_z_std = std::sqrt(state_server.state_cov(14, 14));

    if (position_x_std < position_std_threshold &&
        position_y_std < position_std_threshold &&
        position_z_std < position_std_threshold) {
        return;
    }

    std::cout << "Start %lld online reset procedure..."
        << ++online_reset_counter << std::endl;
    std::cout << "Stardard deviation in xyz: %f, %f, %f"
        << position_x_std << position_y_std << position_z_std << std::endl;

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    // Reset the state covariance.
    state_server.state_cov = Eigen::MatrixXd::Zero(21, 21);
    for (int32_t i = 3; i < 6; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_gryo_bias;
    }
    for (int32_t i = 6; i < 9; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_velocity;
    }
    for (int32_t i = 9; i < 12; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_acc_bias;
    }
    for (int32_t i = 15; i < 18; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_rot;
    }
    for (int32_t i = 18; i < 21; ++i) {
        state_server.state_cov(i, i) = param_initial_covariance_extr_trans;
    }

    std::cout << "online reset complete..." << online_reset_counter << std::endl;
}

void MsckfVio::getFusionResult(
    Odometry_t& odom,
    std::vector<Vector3f_t>& feature_pcl)
{
    odom.stamp = state_server.imu_state.time;

    // Convert the IMU frame to the body frame.
    const IMUState& imu_state = state_server.imu_state;
    Eigen::Isometry3d T_i_w(Eigen::Isometry3d::Identity());
    T_i_w.linear() = quaternionToRotation(
        imu_state.orientation).transpose();
    T_i_w.translation() = imu_state.position;

    Eigen::Isometry3d T_b_w(T_i_w * IMUState::T_imu_body.inverse());
    Eigen::Vector3d body_velocity =
        IMUState::T_imu_body.linear() * imu_state.velocity;
    odom.position.x = (float)T_b_w.translation()[0];
    odom.position.y = (float)T_b_w.translation()[1];
    odom.position.z = (float)T_b_w.translation()[2];

    // Convert the covariance.
    Eigen::Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
    Eigen::Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
    Eigen::Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
    Eigen::Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
    Eigen::Matrix<double, 6, 6> P_imu_pose(Eigen::Matrix<double, 6, 6>::Zero());
    P_imu_pose << P_pp, P_po, P_op, P_oo;

    Eigen::Matrix<double, 6, 6> H_pose(Eigen::Matrix<double, 6, 6>::Zero());
    H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
    H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
    Eigen::Matrix<double, 6, 6> P_body_pose =
        H_pose * P_imu_pose * H_pose.transpose();

    for (int32_t i = 0; i < 6; ++i) {
        for (int32_t j = 0; j < 6; ++j) {
            odom.pose_covariance[6 * i + j] = P_body_pose(i, j);
        }
    }

    // Construct the covariance for the velocity.
    Eigen::Matrix3d P_imu_vel(state_server.state_cov.block<3, 3>(6, 6));
    Eigen::Matrix3d H_vel = IMUState::T_imu_body.linear();
    Eigen::Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
    for (int32_t i = 0; i < 3; ++i) {
        for (int32_t j = 0; j < 3; ++j) {
            odom.twist_covariance[i * 6 + j] = P_body_vel(i, j);
        }
    }

    // 3D positions of the features that has been initialized.
    Eigen::Vector3d feature_position;
    Vector3f_t feature_pt;
    for (const auto& item : map_server) {
        const auto& feature = item.second;
        if (feature.is_initialized) {
            feature_position =
                IMUState::T_imu_body.linear() * feature.position;
            feature_pt.x = (float)feature_position(0);
            feature_pt.y = (float)feature_position(1);
            feature_pt.z = (float)feature_position(2);
            feature_pcl.push_back(feature_pt);
        }
    }
}

} // namespace msckf_vio
