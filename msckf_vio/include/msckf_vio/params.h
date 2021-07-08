#ifndef PARAMS_H
#define PARAMS_H

#include <stdint.h>
#include <string>
#include <vector>

namespace msckf_vio {

/*
* @brief parameters for image process
*/
#define param_grid_row                  (4)
#define param_grid_col                  (4)
#define param_grid_min_feature_num      (2)
#define param_grid_max_feature_num      (4)
#define param_pyramid_levels            (3)
#define param_patch_size                (31)
#define param_fast_threshold            (20)
#define param_max_iteration             (30)
#define param_track_precision           (0.01)
#define param_ransac_threshold          (3)
#define param_stereo_threshold          (3)

/*
* @brief vio parameters
*/
#define param_fixed_frame_id                ("world")
#define param_child_frame_id                ("robot")
#define param_frame_rate                    (40.0)
#define param_position_std_threshold        (8.0)

#define param_rotation_threshold            (0.2618)
#define param_translation_threshold         (0.4)
#define param_tracking_rate_threshold       (0.5)

#define param_feature_translation_threshold (0.2)

#define param_noise_gyro                    (0.001)
#define param_noise_acc                     (0.01)
#define param_noise_gyro_bias               (0.001)
#define param_noise_acc_bias                (0.01)
#define param_noise_feature                 (0.01)

#define param_initial_state_velocity_x      (0.0)
#define param_initial_state_velocity_y      (0.0)
#define param_initial_state_velocity_z      (0.0)

#define param_initial_covariance_velocity   (0.25)
#define param_initial_covariance_gryo_bias  (1e-4)
#define param_initial_covariance_acc_bias   (1e-2)

#define param_initial_covariance_extr_rot   (3.0462e-4)
#define param_initial_covariance_extr_trans (1e-4)

#define param_max_cam_state_size            (30)

/*
* @brief stereo camera intrinsics and extrinsics
*/
typedef struct {
    std::string cam0_distortion_model;
    int32_t cam0_resolution[2];
    double cam0_intrinsics[4];
    double cam0_distortion_coeffs[4];
    double R_cam0_imu[9];
    double t_cam0_imu[3];

    std::string cam1_distortion_model;
    int32_t cam1_resolution[2];
    double cam1_intrinsics[4];
    double cam1_distortion_coeffs[4];
    double R_cam1_imu[9];
    double t_cam1_imu[3];

    double R_imu_body[9];
    double t_imu_body[3];
} Stereo_camera_config_t;

bool loadCaliParameters(
    std::string calib_file,
    Stereo_camera_config_t& stereo_param);

}

#endif
