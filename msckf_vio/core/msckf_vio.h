/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include "msckf_vio/types.h"
#include "msckf_vio/params.h"

#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <deque>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace msckf_vio {

template <typename T>
class MsckfVioStatic
{
public:
    // Chi squared test table.
    static std::map<int32_t, double> chi_squared_test_table;
};

/*
 * @brief MsckfVio Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class MsckfVio : public MsckfVioStatic<void>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    MsckfVio();

    // Disable copy and assign constructor
    MsckfVio(const MsckfVio&) = delete;
    MsckfVio operator=(const MsckfVio&) = delete;

    // Destructor
    ~MsckfVio() {}

    /*
     * @brief initialize Initialize the VIO.
    */
    bool initialize(
        Stereo_camera_config_t& stereo_param);

    bool needInitGravityAndBias() {
        return !is_gravity_set;
    }

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
    */
    void initializeGravityAndBias(
        const std::deque<Sensor_imu_t>& imu_buffer);

    /*
     * @biref reset the VIO to initial status.
     *    Note that this is NOT anytime-reset. This function should
     *    only be called before the sensor suite starts moving.
     *    e.g. while the robot is still on the ground.
    */
    void reset();

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
    */
    void featureCallback(
        const Feature_measure_t& measure,
        const std::deque<Sensor_imu_t>& imu_buffer);

    void getFusionResult(
        Odometry_t& odom,
        std::vector<Vector3f_t>& feature_pcl);

    typedef std::shared_ptr<MsckfVio> Ptr;
    typedef std::shared_ptr<const MsckfVio> ConstPtr;

private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer {
        IMUState imu_state;
        CamStateServer cam_states;

        // State covariance matrix
        Eigen::MatrixXd state_cov;
        Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };

    struct NoiseServer {
        double gyro;
        double acc;
        double gyro_bias;
        double acc_bias;
        double feature;
    };

    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters(Stereo_camera_config_t& params);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const uint64_t& time_bound,
        const std::deque<Sensor_imu_t>& imu_buffer);
    void processModel(const uint64_t& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc);

    // Measurement update
    void stateAugmentation(const uint64_t& time);
    void addFeatureObservations(const Feature_measure_t& measure);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 4, 6>& H_x,
        Eigen::Matrix<double, 4, 3>& H_f,
        Eigen::Vector4d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    bool gatingTest(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const uint32_t& dof);
    void removeLostFeatures();
    void findRedundantCamStates(
        std::vector<StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();

private:
    // State vector
    StateServer state_server;
    // Maximum number of camera states
    int32_t max_cam_state_size;

    // Features used
    MapServer map_server;

    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;

    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;
};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

template <typename T>
std::map<int32_t, double> MsckfVioStatic<T>::chi_squared_test_table;

} // namespace msckf_vio

#endif
