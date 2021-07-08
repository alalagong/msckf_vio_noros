#include "msckf_vio/params.h"

#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace msckf_vio {

bool loadCaliParameters(
    std::string calib_file,
    Stereo_camera_config_t& stereo_param)
{
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout
            << "can't open calibration file: "
            << calib_file << std::endl;
        return false;
    }

    // Camera0 calibration parameters
    cv::FileNode node = fs["cam0"];
    if (node.empty()) {
        stereo_param.cam0_distortion_model = std::string("radtan");
    }
    else {
        node["distortion_model"] >> stereo_param.cam0_distortion_model;
    }

    std::vector<int32_t> cam_resolution_temp(2);
    node["resolution"] >> cam_resolution_temp;
    memcpy(stereo_param.cam0_resolution,
        &cam_resolution_temp[0], sizeof(double)* 2);

    std::vector<double> cam_intrinsics_temp(4);
    node["intrinsics"] >> cam_intrinsics_temp;
    memcpy(stereo_param.cam0_intrinsics,
        &cam_intrinsics_temp[0], sizeof(double)* 4);

    std::vector<double> cam_distortion_coeffs_temp(4);;
    node["distortion_coeffs"] >> cam_distortion_coeffs_temp;
    memcpy(stereo_param.cam0_distortion_coeffs,
        &cam_distortion_coeffs_temp[0], sizeof(double)* 4);

    std::vector<double> T_imu_cam0_temp;
    node["T_cam_imu"] >> T_imu_cam0_temp;
    cv::Mat T_imu_cam0(4, 4, CV_64F, &T_imu_cam0_temp[0]);
    cv::Mat R_imu_cam0 = T_imu_cam0(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat t_imu_cam0 = T_imu_cam0(cv::Rect(3, 0, 1, 3)).clone();
    cv::Mat R_cam0_imu(R_imu_cam0.t());
    cv::Mat t_cam0_imu(-R_cam0_imu * t_imu_cam0);
    memcpy(stereo_param.R_cam0_imu, R_cam0_imu.data, sizeof(double)* 9);
    memcpy(stereo_param.t_cam0_imu, t_cam0_imu.data, sizeof(double)* 3);

    // Camera1 calibration parameters
    node = fs["cam1"];
    if (node.empty()) {
        stereo_param.cam1_distortion_model = std::string("radtan");
    }
    else {
        node["distortion_model"] >> stereo_param.cam1_distortion_model;
    }

    node["resolution"] >> cam_resolution_temp;
    memcpy(stereo_param.cam1_resolution,
        &cam_resolution_temp[0], sizeof(double)* 2);

    node["intrinsics"] >> cam_intrinsics_temp;
    memcpy(stereo_param.cam1_intrinsics,
        &cam_intrinsics_temp[0], sizeof(double)* 4);

    node["distortion_coeffs"] >> cam_distortion_coeffs_temp;
    memcpy(stereo_param.cam1_distortion_coeffs,
        &cam_distortion_coeffs_temp[0], sizeof(double)* 4);

    std::vector<double> T_imu_cam1_temp;
    node["T_cam_imu"] >> T_imu_cam1_temp;
    cv::Mat T_imu_cam1(4, 4, CV_64F, &T_imu_cam1_temp[0]);
    cv::Mat R_imu_cam1 = T_imu_cam1(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat t_imu_cam1 = T_imu_cam1(cv::Rect(3, 0, 1, 3)).clone();
    cv::Mat R_cam1_imu(R_imu_cam1.t());
    cv::Mat t_cam1_imu(-R_cam1_imu * t_imu_cam1);
    memcpy(stereo_param.R_cam1_imu, R_cam1_imu.data, sizeof(double)* 9);
    memcpy(stereo_param.t_cam1_imu, t_cam1_imu.data, sizeof(double)* 3);

    // T_imu_body parameters
    cv::Mat R_imu_body, t_imu_body;
    node = fs["T_imu_body"];
    if (node.empty()) {
        R_imu_body = cv::Mat::eye(3, 3, CV_64F);
        t_imu_body = cv::Mat::zeros(3, 1, CV_64F);
    }
    else {
        std::vector<double> T_imu_body_temp;
        node >> T_imu_body_temp;
        cv::Mat T_imu_body(4, 4, CV_64F, &T_imu_body_temp[0]);
        R_imu_body = T_imu_body(cv::Rect(0, 0, 3, 3)).clone();
        t_imu_body = T_imu_body(cv::Rect(3, 0, 1, 3)).clone();
    }
    memcpy(stereo_param.R_imu_body, R_imu_body.data, sizeof(double)* 9);
    memcpy(stereo_param.t_imu_body, t_imu_body.data, sizeof(double)* 3);

    return true;
}

}