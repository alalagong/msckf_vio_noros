/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include "image_processor.h"

#include <iostream>
#include <algorithm>
#include <stdlib.h>

#include <Eigen/Dense>

namespace msckf_vio {

ImageProcessor::ImageProcessor() :
    is_first_img(true),
    next_feature_id(0),
    prev_features_ptr(new GridFeatures()),
    curr_features_ptr(new GridFeatures())
{
}

ImageProcessor::~ImageProcessor()
{
    cv::destroyAllWindows();
    return;
}

bool ImageProcessor::initialize(
    Stereo_camera_config_t& params)
{
    if (!loadParameters(params)) {
        return false;
    }
    std::cout << "Finish loading parameters..." << std::endl;

    // Create feature detector.
    detector_ptr = cv::FastFeatureDetector::create(
        processor_config.fast_threshold);

    return true;
}

bool ImageProcessor::loadParameters(
    Stereo_camera_config_t& params)
{
    // Camera0 calibration parameters
    cam0_distortion_model = params.cam0_distortion_model;

    cam0_resolution = cv::Vec2i(
        params.cam0_resolution[0],
        params.cam0_resolution[1]);

    cam0_intrinsics = cv::Vec4d(
        params.cam0_intrinsics[0],
        params.cam0_intrinsics[1],
        params.cam0_intrinsics[2],
        params.cam0_intrinsics[3]);

    cam0_distortion_coeffs = cv::Vec4d(
        params.cam0_distortion_coeffs[0],
        params.cam0_distortion_coeffs[1],
        params.cam0_distortion_coeffs[2],
        params.cam0_distortion_coeffs[3]);

    R_cam0_imu = cv::Matx33d(
        params.R_cam0_imu[0], params.R_cam0_imu[1], params.R_cam0_imu[2],
        params.R_cam0_imu[3], params.R_cam0_imu[4], params.R_cam0_imu[5],
        params.R_cam0_imu[6], params.R_cam0_imu[7], params.R_cam0_imu[8]);

    t_cam0_imu = cv::Vec3d(
        params.t_cam0_imu[0],
        params.t_cam0_imu[1],
        params.t_cam0_imu[2]);

    // Camera1 calibration parameters
    cam1_distortion_model = params.cam1_distortion_model;

    cam1_resolution = cv::Vec2i(
        params.cam1_resolution[0],
        params.cam1_resolution[1]);

    cam1_intrinsics = cv::Vec4d(
        params.cam1_intrinsics[0],
        params.cam1_intrinsics[1],
        params.cam1_intrinsics[2],
        params.cam1_intrinsics[3]);

    cam1_distortion_coeffs = cv::Vec4d(
        params.cam1_distortion_coeffs[0],
        params.cam1_distortion_coeffs[1],
        params.cam1_distortion_coeffs[2],
        params.cam1_distortion_coeffs[3]);

    R_cam1_imu = cv::Matx33d(
        params.R_cam1_imu[0], params.R_cam1_imu[1], params.R_cam1_imu[2],
        params.R_cam1_imu[3], params.R_cam1_imu[4], params.R_cam1_imu[5],
        params.R_cam1_imu[6], params.R_cam1_imu[7], params.R_cam1_imu[8]);

    t_cam1_imu = cv::Vec3d(
        params.t_cam1_imu[0],
        params.t_cam1_imu[1],
        params.t_cam1_imu[2]);

    // Processor parameters
    processor_config.grid_row = param_grid_row;
    processor_config.grid_col = param_grid_col;
    processor_config.grid_min_feature_num = param_grid_min_feature_num;
    processor_config.grid_max_feature_num = param_grid_max_feature_num;
    processor_config.pyramid_levels = param_pyramid_levels;
    processor_config.patch_size = param_patch_size;
    processor_config.fast_threshold = param_fast_threshold;
    processor_config.max_iteration = param_max_iteration;
    processor_config.track_precision = param_track_precision;
    processor_config.ransac_threshold = param_ransac_threshold;
    processor_config.stereo_threshold = param_stereo_threshold;
    return true;
}

void ImageProcessor::stereoCallback(
    const Gray_img_t& cam0,
    const Gray_img_t& cam1,
    const std::deque<Sensor_imu_t>& imu_buffer)
{
    // Update the previous image and previous features.
    cam0_prev_img = cam0_curr_img;
    prev_features_ptr.swap(curr_features_ptr);
    std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

    // Get the current image.
    cam0_curr_img.stamp = cam0.stamp;
    cam0_curr_img.img = cv::Mat(
        cam0.rows, cam0.cols, CV_8UC1, cam0.ptr);

    cam1_curr_img.stamp = cam1.stamp;
    cam1_curr_img.img = cv::Mat(
        cam1.rows, cam1.cols, CV_8UC1, cam1.ptr);

    // Build the image pyramids once since they're used at multiple places
    createImagePyramids();

    // Initialize the current features to empty vectors.
    curr_features_ptr->clear();

    // Detect features in the first frame.
    if (is_first_img) {
        initializeFirstFrame();
        is_first_img = false;
    }
    else {
        // Track the feature in the previous image.
        trackFeatures(imu_buffer);

        // Add new features into the current image.
        addNewFeatures();

        // Add new features into the current image.
        pruneGridFeatures();
    }

    //updateFeatureLifetime();
}

void ImageProcessor::createImagePyramids()
{
    buildOpticalFlowPyramid(
        cam0_curr_img.img, curr_cam0_pyramid_,
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels, true, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);

    buildOpticalFlowPyramid(
        cam1_curr_img.img, curr_cam1_pyramid_,
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels, true, cv::BORDER_REFLECT_101,
        cv::BORDER_CONSTANT, false);
}

void ImageProcessor::initializeFirstFrame()
{
    // Size of each grid.
    const int32_t grid_height = cam0_curr_img.img.rows / processor_config.grid_row;
    const int32_t grid_width = cam0_curr_img.img.cols / processor_config.grid_col;

    // Detect new features on the frist image.
    std::vector<cv::KeyPoint> new_features(0);
    detector_ptr->detect(cam0_curr_img.img, new_features);

    // Find the stereo matched points for the newly
    // detected features.
    std::vector<cv::Point2f> cam0_points(new_features.size());
    for (uint32_t i = 0; i < new_features.size(); ++i) {
        cam0_points[i] = new_features[i].pt;
    }

    std::vector<cv::Point2f> cam1_points(0);
    std::vector<uint8_t> inlier_markers(0);
    stereoMatch(cam0_points, cam1_points, inlier_markers);

    std::vector<cv::Point2f> cam0_inliers(0);
    std::vector<cv::Point2f> cam1_inliers(0);
    std::vector<float> response_inliers(0);
    for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
        if (inlier_markers[i] == 0) {
            continue;
        }
        cam0_inliers.push_back(cam0_points[i]);
        cam1_inliers.push_back(cam1_points[i]);
        response_inliers.push_back(new_features[i].response);
    }

    // Group the features into grids
    GridFeatures grid_new_features;
    for (uint32_t i = 0; i < cam0_inliers.size(); ++i) {
        const cv::Point2f& cam0_point = cam0_inliers[i];
        const cv::Point2f& cam1_point = cam1_inliers[i];
        const float& response = response_inliers[i];

        const int32_t row = static_cast<int32_t>(cam0_point.y / grid_height);
        const int32_t col = static_cast<int32_t>(cam0_point.x / grid_width);
        const int32_t code = row * processor_config.grid_col + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        new_feature.cam1_point = cam1_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    for (auto& item : grid_new_features) {
        std::sort(item.second.begin(), item.second.end(),
            &ImageProcessor::featureCompareByResponse);
    }

    // Collect new features within each grid with high response.
    const int32_t grid_cnt = processor_config.grid_row * processor_config.grid_col;
    for (int32_t code = 0; code < grid_cnt; ++code) {
        std::vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
        std::vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

        for (int32_t k = 0; k < processor_config.grid_min_feature_num &&
            k < (int32_t)new_features_this_grid.size(); ++k) {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;
        }
    }
}

void ImageProcessor::predictFeatureTracking(
    const std::vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    std::vector<cv::Point2f>& compensated_pts)
{
    // Return directly if there are no input features.
    if (input_pts.size() == 0) {
        compensated_pts.clear();
        return;
    }
    compensated_pts.resize(input_pts.size());

    // Intrinsic matrix.
    const cv::Matx33f K(
        (float)intrinsics[0], 0.0f, (float)intrinsics[2],
        0.0f, (float)intrinsics[1], (float)intrinsics[3],
        0.0f, 0.0f, 1.0f);
    const cv::Matx33f H = K * R_p_c * K.inv();

    for (uint32_t i = 0; i < input_pts.size(); ++i) {
        const cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
        const cv::Vec3f p2 = H * p1;
        compensated_pts[i].x = p2[0] / p2[2];
        compensated_pts[i].y = p2[1] / p2[2];
    }
}

void ImageProcessor::trackFeatures(
    const std::deque<Sensor_imu_t>& imu_buffer)
{
    // Size of each grid.
    const int32_t grid_height =
        cam0_curr_img.img.rows / processor_config.grid_row;
    const int32_t grid_width =
        cam0_curr_img.img.cols / processor_config.grid_col;

    // Compute a rough relative rotation which takes a vector
    // from the previous frame to the current frame.
    cv::Matx33f cam0_R_p_c;
    cv::Matx33f cam1_R_p_c;
    integrateImuData(cam0_R_p_c, cam1_R_p_c, imu_buffer);

    // Organize the features in the previous image.
    std::vector<FeatureIDType> prev_ids(0);
    std::vector<int32_t> prev_lifetime(0);
    std::vector<cv::Point2f> prev_cam0_points(0);
    std::vector<cv::Point2f> prev_cam1_points(0);

    for (const auto& item : *prev_features_ptr) {
        for (const auto& prev_feature : item.second) {
            prev_ids.push_back(prev_feature.id);
            prev_lifetime.push_back(prev_feature.lifetime);
            prev_cam0_points.push_back(prev_feature.cam0_point);
            prev_cam1_points.push_back(prev_feature.cam1_point);
        }
    }

    // Number of the features before tracking.
    before_tracking = (uint32_t)prev_cam0_points.size();

    // Abort tracking if there is no features in
    // the previous frame.
    if (prev_ids.size() == 0) {
        return;
    }

    // Track features using LK optical flow method.
    std::vector<cv::Point2f> curr_cam0_points(0);
    std::vector<uint8_t> track_inliers(0);

    predictFeatureTracking(prev_cam0_points,
        cam0_R_p_c, cam0_intrinsics, curr_cam0_points);

    calcOpticalFlowPyrLK(
        prev_cam0_pyramid_, curr_cam0_pyramid_,
        prev_cam0_points, curr_cam0_points,
        track_inliers, cv::noArray(),
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        processor_config.max_iteration,
        processor_config.track_precision),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (uint32_t i = 0; i < curr_cam0_points.size(); ++i) {
        if (track_inliers[i] == 0) {
            continue;
        }
        if (curr_cam0_points[i].y < 0 ||
            curr_cam0_points[i].y > cam0_curr_img.img.rows - 1 ||
            curr_cam0_points[i].x < 0 ||
            curr_cam0_points[i].x > cam0_curr_img.img.cols - 1) {
            track_inliers[i] = 0;
        }
    }

    // Collect the tracked points.
    std::vector<FeatureIDType> prev_tracked_ids(0);
    std::vector<int32_t> prev_tracked_lifetime(0);
    std::vector<cv::Point2f> prev_tracked_cam0_points(0);
    std::vector<cv::Point2f> prev_tracked_cam1_points(0);
    std::vector<cv::Point2f> curr_tracked_cam0_points(0);

    removeUnmarkedElements(
        prev_ids, track_inliers, prev_tracked_ids);
    removeUnmarkedElements(
        prev_lifetime, track_inliers, prev_tracked_lifetime);
    removeUnmarkedElements(
        prev_cam0_points, track_inliers, prev_tracked_cam0_points);
    removeUnmarkedElements(
        prev_cam1_points, track_inliers, prev_tracked_cam1_points);
    removeUnmarkedElements(
        curr_cam0_points, track_inliers, curr_tracked_cam0_points);

    // Number of features left after tracking.
    after_tracking = (uint32_t)curr_tracked_cam0_points.size();

    // Outlier removal involves three steps, which forms a close
    // loop between the previous and current frames of cam0 (left)
    // and cam1 (right). Assuming the stereo matching between the
    // previous cam0 and cam1 images are correct, the three steps are:
    //
    // prev frames cam0 ----------> cam1
    //              |                |
    //              |ransac          |ransac
    //              |  stereo match  |
    // curr frames cam0 ----------> cam1
    //
    // 1) Stereo matching between current images of cam0 and cam1.
    // 2) RANSAC between previous and current images of cam0.
    // 3) RANSAC between previous and current images of cam1.
    //
    // For Step 3, tracking between the images is no longer needed.
    // The stereo matching results are directly used in the RANSAC.

    // Step 1: stereo matching.
    std::vector<cv::Point2f> curr_cam1_points(0);
    std::vector<uint8_t> match_inliers(0);
    stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

    std::vector<FeatureIDType> prev_matched_ids(0);
    std::vector<int32_t> prev_matched_lifetime(0);
    std::vector<cv::Point2f> prev_matched_cam0_points(0);
    std::vector<cv::Point2f> prev_matched_cam1_points(0);
    std::vector<cv::Point2f> curr_matched_cam0_points(0);
    std::vector<cv::Point2f> curr_matched_cam1_points(0);

    removeUnmarkedElements(
        prev_tracked_ids, match_inliers, prev_matched_ids);
    removeUnmarkedElements(
        prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
    removeUnmarkedElements(
        prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
    removeUnmarkedElements(
        prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
    removeUnmarkedElements(
        curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
    removeUnmarkedElements(
        curr_cam1_points, match_inliers, curr_matched_cam1_points);

    // Number of features left after stereo matching.
    after_matching = (uint32_t)curr_matched_cam0_points.size();

    // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
    std::vector<int32_t> cam0_ransac_inliers(0);
    twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
        cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
        cam0_distortion_coeffs, processor_config.ransac_threshold,
        0.99, cam0_ransac_inliers);

    std::vector<int32_t> cam1_ransac_inliers(0);
    twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
        cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
        cam1_distortion_coeffs, processor_config.ransac_threshold,
        0.99, cam1_ransac_inliers);

    // Number of features after ransac.
    after_ransac = 0;

    for (uint32_t i = 0; i < cam0_ransac_inliers.size(); ++i) {
        if (cam0_ransac_inliers[i] == 0 ||
            cam1_ransac_inliers[i] == 0) {
            continue;
        }
        int32_t row = static_cast<int32_t>(
            curr_matched_cam0_points[i].y / grid_height);
        int32_t col = static_cast<int32_t>(
            curr_matched_cam0_points[i].x / grid_width);
        int32_t code = row * processor_config.grid_col + col;
        (*curr_features_ptr)[code].push_back(FeatureMetaData());

        FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
        grid_new_feature.id = prev_matched_ids[i];
        grid_new_feature.lifetime = ++prev_matched_lifetime[i];
        grid_new_feature.cam0_point = curr_matched_cam0_points[i];
        grid_new_feature.cam1_point = curr_matched_cam1_points[i];

        ++after_ransac;
    }

    // Compute the tracking rate.
    uint32_t prev_feature_num = 0;
    for (const auto& item : *prev_features_ptr) {
        prev_feature_num += (uint32_t)item.second.size();
    }

    uint32_t curr_feature_num = 0;
    for (const auto& item : *curr_features_ptr) {
        curr_feature_num += (uint32_t)item.second.size();
    }

    std::cout << "candidates: " << before_tracking << "; "
        << "raw track:" << after_tracking << "; "
        << "stereo match: " << after_matching << "; "
        << "ransac: " << curr_feature_num << "/" << prev_feature_num
        << "=" << (double)(curr_feature_num) / ((double)(prev_feature_num)+1e-5)
        << std::endl;
}

void ImageProcessor::stereoMatch(
    const std::vector<cv::Point2f>& cam0_points,
    std::vector<cv::Point2f>& cam1_points,
    std::vector<uint8_t>& inlier_markers) {

    if (cam0_points.size() == 0) {
        return;
    }

    if (cam1_points.size() == 0) {
        // Initialize cam1_points by projecting cam0_points to cam1 using the
        // rotation from stereo extrinsics
        const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
        std::vector<cv::Point2f> cam0_points_undistorted;
        undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,
                        cam0_distortion_coeffs, cam0_points_undistorted,
                        R_cam0_cam1);
        cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics,
                                    cam1_distortion_model, cam1_distortion_coeffs);
    }

    // Track features using LK optical flow method.
    calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_,
        cam0_points, cam1_points,
        inlier_markers, cv::noArray(),
        cv::Size(processor_config.patch_size, processor_config.patch_size),
        processor_config.pyramid_levels,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                        processor_config.max_iteration,
                        processor_config.track_precision),
                        cv::OPTFLOW_USE_INITIAL_FLOW);

    // Mark those tracked points out of the image region
    // as untracked.
    for (uint32_t i = 0; i < cam1_points.size(); ++i) {
        if (inlier_markers[i] == 0) {
            continue;
        }
        if (cam1_points[i].y < 0 ||
            cam1_points[i].y > cam1_curr_img.img.rows - 1 ||
            cam1_points[i].x < 0 ||
            cam1_points[i].x > cam1_curr_img.img.cols - 1) {
            inlier_markers[i] = 0;
        }
    }

    // Compute the relative rotation between the cam0
    // frame and cam1 frame.
    const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
    const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu - t_cam1_imu);
    // Compute the essential matrix.
    const cv::Matx33d t_cam0_cam1_hat(
        0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
        t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
        -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
    const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

    // Further remove outliers based on the known
    // essential matrix.
    std::vector<cv::Point2f> cam0_points_undistorted(0);
    std::vector<cv::Point2f> cam1_points_undistorted(0);
    undistortPoints(
        cam0_points, cam0_intrinsics, cam0_distortion_model,
        cam0_distortion_coeffs, cam0_points_undistorted);
    undistortPoints(
        cam1_points, cam1_intrinsics, cam1_distortion_model,
        cam1_distortion_coeffs, cam1_points_undistorted);

    double norm_pixel_unit = 4.0 / (
        cam0_intrinsics[0] + cam0_intrinsics[1] +
        cam1_intrinsics[0] + cam1_intrinsics[1]);

    for (uint32_t i = 0; i < cam0_points_undistorted.size(); ++i) {
        if (inlier_markers[i] == 0) {
            continue;
        }
        cv::Vec3d pt0(cam0_points_undistorted[i].x,
            cam0_points_undistorted[i].y, 1.0);
        cv::Vec3d pt1(cam1_points_undistorted[i].x,
            cam1_points_undistorted[i].y, 1.0);
        cv::Vec3d epipolar_line = E * pt0;
        double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(
            epipolar_line[0] * epipolar_line[0] +
            epipolar_line[1] * epipolar_line[1]);
        if (error > processor_config.stereo_threshold * norm_pixel_unit) {
            inlier_markers[i] = 0;
        }
    }
}

void ImageProcessor::addNewFeatures()
{
    // Size of each grid.
    static int32_t grid_height =
        cam0_curr_img.img.rows / processor_config.grid_row;
    static int32_t grid_width =
        cam0_curr_img.img.cols / processor_config.grid_col;

    // Create a mask to avoid redetecting existing features.
    cv::Mat mask(cam0_curr_img.img.rows,
        cam0_curr_img.img.cols, CV_8U, cv::Scalar(1));

    for (const auto& features : *curr_features_ptr) {
        for (const auto& feature : features.second) {
            const int32_t y = static_cast<int32_t>(feature.cam0_point.y);
            const int32_t x = static_cast<int32_t>(feature.cam0_point.x);

            int32_t up_lim = y - 2, bottom_lim = y + 3,
                left_lim = x - 2, right_lim = x + 3;
            if (up_lim < 0) {
                up_lim = 0;
            }
            if (bottom_lim > cam0_curr_img.img.rows) {
                bottom_lim = cam0_curr_img.img.rows;
            }
            if (left_lim < 0) {
                left_lim = 0;
            }
            if (right_lim > cam0_curr_img.img.cols) {
                right_lim = cam0_curr_img.img.cols;
            }

            cv::Range row_range(up_lim, bottom_lim);
            cv::Range col_range(left_lim, right_lim);
            mask(row_range, col_range) = 0;
        }
    }

    // Detect new features.
    std::vector<cv::KeyPoint> new_features(0);
    detector_ptr->detect(cam0_curr_img.img, new_features, mask);

    // Collect the new detected features based on the grid.
    // Select the ones with top response within each grid afterwards.
    std::vector<std::vector<cv::KeyPoint> > new_feature_sieve(
        processor_config.grid_row * processor_config.grid_col);
    for (const auto& feature : new_features) {
        int32_t row = static_cast<int32_t>(feature.pt.y / grid_height);
        int32_t col = static_cast<int32_t>(feature.pt.x / grid_width);
        new_feature_sieve[
            row * processor_config.grid_col + col].push_back(feature);
    }

    new_features.clear();
    for (auto& item : new_feature_sieve) {
        if ((int32_t)item.size() > processor_config.grid_max_feature_num) {
            std::sort(item.begin(), item.end(),
                &ImageProcessor::keyPointCompareByResponse);
            item.erase(
                item.begin() + processor_config.grid_max_feature_num,
                item.end());
        }
        new_features.insert(new_features.end(), item.begin(), item.end());
    }

    uint32_t detected_new_features = (uint32_t)new_features.size();

    // Find the stereo matched points for the newly
    // detected features.
    std::vector<cv::Point2f> cam0_points(new_features.size());
    for (uint32_t i = 0; i < new_features.size(); ++i) {
        cam0_points[i] = new_features[i].pt;
    }

    std::vector<cv::Point2f> cam1_points(0);
    std::vector<uint8_t> inlier_markers(0);
    stereoMatch(cam0_points, cam1_points, inlier_markers);

    std::vector<cv::Point2f> cam0_inliers(0);
    std::vector<cv::Point2f> cam1_inliers(0);
    std::vector<float> response_inliers(0);
    for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
        if (inlier_markers[i] == 0) {
            continue;
        }
        cam0_inliers.push_back(cam0_points[i]);
        cam1_inliers.push_back(cam1_points[i]);
        response_inliers.push_back(new_features[i].response);
    }

    uint32_t matched_new_features = (uint32_t)cam0_inliers.size();

    if (matched_new_features < 5 &&
        static_cast<double>(matched_new_features) /
        static_cast<double>(detected_new_features) < 0.1) {
        //printf("Images at %lld seems unsynced...\n",
        //    cam0_curr_img.stamp);
    }

    // Group the features into grids
    GridFeatures grid_new_features;
    for (uint32_t i = 0; i < cam0_inliers.size(); ++i) {
        const cv::Point2f& cam0_point = cam0_inliers[i];
        const cv::Point2f& cam1_point = cam1_inliers[i];
        const float& response = response_inliers[i];

        int32_t row = static_cast<int32_t>(cam0_point.y / grid_height);
        int32_t col = static_cast<int32_t>(cam0_point.x / grid_width);
        int32_t code = row * processor_config.grid_col + col;

        FeatureMetaData new_feature;
        new_feature.response = response;
        new_feature.cam0_point = cam0_point;
        new_feature.cam1_point = cam1_point;
        grid_new_features[code].push_back(new_feature);
    }

    // Sort the new features in each grid based on its response.
    for (auto& item : grid_new_features) {
        std::sort(item.second.begin(), item.second.end(),
            &ImageProcessor::featureCompareByResponse);
    }

    int32_t new_added_feature_num = 0;
    // Collect new features within each grid with high response.
    for (int32_t code = 0; code <
        processor_config.grid_row*processor_config.grid_col; ++code) {
        std::vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
        std::vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

        if ((int32_t)features_this_grid.size() >=
            processor_config.grid_min_feature_num) {
            continue;
        }

        int32_t vacancy_num = processor_config.grid_min_feature_num -
            (int32_t)features_this_grid.size();
        for (int32_t k = 0;
            k < vacancy_num && k < (int32_t)new_features_this_grid.size(); ++k) {
            features_this_grid.push_back(new_features_this_grid[k]);
            features_this_grid.back().id = next_feature_id++;
            features_this_grid.back().lifetime = 1;

            ++new_added_feature_num;
        }
    }

    //std::cout << "detected: " << detected_new_features << "; "
    //    << "matched: " << matched_new_features << "; "
    //    << "new added feature: " << new_added_feature_num
    //    << std::endl;
}

void ImageProcessor::pruneGridFeatures()
{
    for (auto& item : *curr_features_ptr) {
        auto& grid_features = item.second;
        // Continue if the number of features in this grid does
        // not exceed the upper bound.
        if ((int32_t)grid_features.size() <=
            processor_config.grid_max_feature_num) {
            continue;
        }
        std::sort(grid_features.begin(), grid_features.end(),
            &ImageProcessor::featureCompareByLifetime);
        grid_features.erase(grid_features.begin()+
            processor_config.grid_max_feature_num,
            grid_features.end());
    }
    return;
}

void ImageProcessor::undistortPoints(
    const std::vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    std::vector<cv::Point2f>& pts_out,
    const cv::Matx33d& rectification_matrix,
    const cv::Vec4d& new_intrinsics)
{

    if (pts_in.size() == 0) {
        return;
    }

    const cv::Matx33d K(
        intrinsics[0], 0.0, intrinsics[2],
        0.0, intrinsics[1], intrinsics[3],
        0.0, 0.0, 1.0);

    const cv::Matx33d K_new(
        new_intrinsics[0], 0.0, new_intrinsics[2],
        0.0, new_intrinsics[1], new_intrinsics[3],
        0.0, 0.0, 1.0);

    if (distortion_model == "radtan") {
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);
    } else if (distortion_model == "equidistant") {
        cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);
    } else {
        std::cout << "The model %s is unrecognized, use radtan instead..."
                  << distortion_model.c_str() << std::endl;
        cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                            rectification_matrix, K_new);
    }
}

std::vector<cv::Point2f> ImageProcessor::distortPoints(
    const std::vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs)
{
    const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                        0.0, intrinsics[1], intrinsics[3],
                        0.0, 0.0, 1.0);

    std::vector<cv::Point2f> pts_out;
    if (distortion_model == "radtan") {
        std::vector<cv::Point3f> homogenous_pts;
        cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
        cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                            distortion_coeffs, pts_out);
    } else if (distortion_model == "equidistant") {
        cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
    } else {
        std::cout << "The model %s is unrecognized, using radtan instead..."
                  << distortion_model.c_str() << std::endl;
        std::vector<cv::Point3f> homogenous_pts;
        cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
        cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                        distortion_coeffs, pts_out);
    }

    return pts_out;
}

void ImageProcessor::integrateImuData(
    cv::Matx33f& cam0_R_p_c,
    cv::Matx33f& cam1_R_p_c,
    const std::deque<Sensor_imu_t> &imu_buffer)
{
    // Compute the mean angular velocity in the IMU frame.
    cv::Vec3f mean_ang_vel(0.0f, 0.0f, 0.0f);
    uint32_t mean_cnt = 0;

    auto it = imu_buffer.begin();
    auto itend = imu_buffer.end();
    for (; it != itend; it++) {
        if (it->stamp > cam0_curr_img.stamp) {
            break;
        }
        if (it->stamp > cam0_prev_img.stamp) {
            mean_ang_vel += cv::Vec3f(
                it->angular_velocity.x,
                it->angular_velocity.y,
                it->angular_velocity.z);
            mean_cnt++;
        }
    }
    if (mean_cnt > 0) {
        mean_ang_vel /= (float)mean_cnt;
    }

    // Transform the mean angular velocity from the IMU
    // frame to the cam0 and cam1 frames.
    cv::Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
    cv::Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

    // Compute the relative rotation.
    float dtime = (float)(((double)cam0_curr_img.stamp
        - (double)cam0_prev_img.stamp) / 1e9);
    Rodrigues(cam0_mean_ang_vel * dtime, cam0_R_p_c);
    Rodrigues(cam1_mean_ang_vel * dtime, cam1_R_p_c);
    cam0_R_p_c = cam0_R_p_c.t();
    cam1_R_p_c = cam1_R_p_c.t();
}

void ImageProcessor::rescalePoints(
    std::vector<cv::Point2f>& pts1,
    std::vector<cv::Point2f>& pts2,
    float& scaling_factor)
{
    scaling_factor = 0.0f;

    for (uint32_t i = 0; i < pts1.size(); ++i) {
        scaling_factor += sqrt(pts1[i].dot(pts1[i]));
        scaling_factor += sqrt(pts2[i].dot(pts2[i]));
    }

    scaling_factor = (pts1.size() + pts2.size()) /
        scaling_factor * sqrt(2.0f);

    for (uint32_t i = 0; i < pts1.size(); ++i) {
        pts1[i] *= scaling_factor;
        pts2[i] *= scaling_factor;
    }
}

void ImageProcessor::twoPointRansac(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    const double& inlier_error,
    const double& success_probability,
    std::vector<int32_t>& inlier_markers)
{
    // Check the size of input point size.
    if (pts1.size() != pts2.size()) {
        std::cout << "Sets of different size (%lu and %lu) are used..."
            << pts1.size() << pts2.size() << std::endl;
    }

    double norm_pixel_unit = 2.0 / (intrinsics[0] + intrinsics[1]);
    int32_t iter_num = static_cast<int32_t>(
        ceil(log(1 - success_probability) / log(1 - 0.7 * 0.7)));

    // Initially, mark all points as inliers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 1);

    // Undistort all the points.
    std::vector<cv::Point2f> pts1_undistorted(pts1.size());
    std::vector<cv::Point2f> pts2_undistorted(pts2.size());
    undistortPoints(
        pts1, intrinsics, distortion_model,
        distortion_coeffs, pts1_undistorted);
    undistortPoints(
        pts2, intrinsics, distortion_model,
        distortion_coeffs, pts2_undistorted);

    // Compenstate the points in the previous image with
    // the relative rotation.
    for (uint32_t i = 0; i < pts1_undistorted.size(); i++) {
        const cv::Vec3f pt_h(
            pts1_undistorted[i].x,
            pts1_undistorted[i].y, 1.0f);
        //cv::Vec3f pt_hc = dR * pt_h;
        const cv::Vec3f pt_hc = R_p_c * pt_h;
        pts1_undistorted[i].x = pt_hc[0];
        pts1_undistorted[i].y = pt_hc[1];
    }

    // Normalize the points to gain numerical stability.
    float scaling_factor = 0.0f;
    rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
    norm_pixel_unit *= scaling_factor;

    // Compute the difference between previous and current points,
    // which will be used frequently later.
    std::vector<cv::Point2d> pts_diff(pts1_undistorted.size());
    for (uint32_t i = 0; i < pts1_undistorted.size(); ++i) {
        pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];
    }

    // Mark the point pairs with large difference directly.
    // BTW, the mean distance of the rest of the point pairs
    // are computed.
    double mean_pt_distance = 0.0;
    int32_t raw_inlier_cntr = 0;
    for (uint32_t i = 0; i < pts_diff.size(); ++i) {
        double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
        // 25 pixel distance is a pretty large tolerance for normal motion.
        // However, to be used with aggressive motion, this tolerance should
        // be increased significantly to match the usage.
        if (distance > 50.0 * norm_pixel_unit) {
            inlier_markers[i] = 0;
        } else {
            mean_pt_distance += distance;
            ++raw_inlier_cntr;
        }
    }
    mean_pt_distance /= raw_inlier_cntr;

    // If the current number of inliers is less than 3, just mark
    // all input as outliers. This case can happen with fast
    // rotation where very few features are tracked.
    if (raw_inlier_cntr < 3) {
        for (auto& marker : inlier_markers) {
            marker = 0;
        }
        return;
    }

    // Before doing 2-point RANSAC, we have to check if the motion
    // is degenerated, meaning that there is no translation between
    // the frames, in which case, the model of the RANSAC does not
    // work. If so, the distance between the matched points will
    // be almost 0.
    //if (mean_pt_distance < inlier_error * norm_pixel_unit) {
    if (mean_pt_distance < norm_pixel_unit) {
        //ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
        for (uint32_t i = 0; i < pts_diff.size(); ++i) {
            if (inlier_markers[i] == 0) {
                continue;
            }
            if (sqrt(pts_diff[i].dot(pts_diff[i])) >
                inlier_error * norm_pixel_unit)
            inlier_markers[i] = 0;
        }
        return;
    }

    // In the case of general motion, the RANSAC model can be applied.
    // The three column corresponds to tx, ty, and tz respectively.
    Eigen::MatrixXd coeff_t(pts_diff.size(), 3);
    for (uint32_t i = 0; i < pts_diff.size(); ++i) {
        coeff_t(i, 0) = pts_diff[i].y;
        coeff_t(i, 1) = -pts_diff[i].x;
        coeff_t(i, 2) = pts1_undistorted[i].x * pts2_undistorted[i].y -
            pts1_undistorted[i].y * pts2_undistorted[i].x;
    }

    std::vector<int32_t> raw_inlier_idx;
    for (uint32_t i = 0; i < inlier_markers.size(); ++i) {
        if (inlier_markers[i] != 0) {
            raw_inlier_idx.push_back(i);
        }
    }

    std::vector<int32_t> best_inlier_set;
    double best_error = 1e10;
    srand(1);

    for (int32_t iter_idx = 0; iter_idx < iter_num; ++iter_idx) {
        // Randomly select two point pairs.
        // Although this is a weird way of selecting two pairs, but it
        // is able to efficiently avoid selecting repetitive pairs.
        int32_t select_idx1 = rand() % (raw_inlier_idx.size() - 1);
        int32_t select_idx_diff = rand() % (raw_inlier_idx.size() - 2) + 1;
        int32_t select_idx2 = select_idx1 +
            select_idx_diff < (int32_t)raw_inlier_idx.size() ?
            select_idx1 + select_idx_diff :
            select_idx1 + select_idx_diff - (int32_t)raw_inlier_idx.size();

        int32_t pair_idx1 = raw_inlier_idx[select_idx1];
        int32_t pair_idx2 = raw_inlier_idx[select_idx2];

        // Construct the model;
        Eigen::Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
        Eigen::Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
        Eigen::Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
        std::vector<double> coeff_l1_norm(3);
        coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
        coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
        coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
        int32_t base_indicator = (int32_t)(min_element(coeff_l1_norm.begin(),
            coeff_l1_norm.end()) - coeff_l1_norm.begin());

        Eigen::Vector3d model(0.0, 0.0, 0.0);
        if (base_indicator == 0) {
            Eigen::Matrix2d A;
            A << coeff_ty, coeff_tz;
            Eigen::Vector2d solution = A.inverse() * (-coeff_tx);
            model(0) = 1.0;
            model(1) = solution(0);
            model(2) = solution(1);
        } else if (base_indicator ==1) {
            Eigen::Matrix2d A;
            A << coeff_tx, coeff_tz;
            Eigen::Vector2d solution = A.inverse() * (-coeff_ty);
            model(0) = solution(0);
            model(1) = 1.0;
            model(2) = solution(1);
        } else {
            Eigen::Matrix2d A;
            A << coeff_tx, coeff_ty;
            Eigen::Vector2d solution = A.inverse() * (-coeff_tz);
            model(0) = solution(0);
            model(1) = solution(1);
            model(2) = 1.0;
        }

        // Find all the inliers among point pairs.
        Eigen::VectorXd error = coeff_t * model;

        std::vector<int32_t> inlier_set;
        for (int32_t i = 0; i < error.rows(); ++i) {
            if (inlier_markers[i] == 0) {
                continue;
            }
            if (std::abs(error(i)) < inlier_error * norm_pixel_unit) {
                inlier_set.push_back(i);
            }
        }

        // If the number of inliers is small, the current
        // model is probably wrong.
        if (inlier_set.size() < 0.2f * pts1_undistorted.size()) {
            continue;
        }

        // Refit the model using all of the possible inliers.
        Eigen::VectorXd coeff_tx_better(inlier_set.size());
        Eigen::VectorXd coeff_ty_better(inlier_set.size());
        Eigen::VectorXd coeff_tz_better(inlier_set.size());
        for (uint32_t i = 0; i < inlier_set.size(); ++i) {
            coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
            coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
            coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
        }

        Eigen::Vector3d model_better(0.0, 0.0, 0.0);
        if (base_indicator == 0) {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_ty_better, coeff_tz_better;
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
            model_better(0) = 1.0;
            model_better(1) = solution(0);
            model_better(2) = solution(1);
        } else if (base_indicator ==1) {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_tz_better;
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
            model_better(0) = solution(0);
            model_better(1) = 1.0;
            model_better(2) = solution(1);
        } else {
            Eigen::MatrixXd A(inlier_set.size(), 2);
            A << coeff_tx_better, coeff_ty_better;
            Eigen::Vector2d solution =
                (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
            model_better(0) = solution(0);
            model_better(1) = solution(1);
            model_better(2) = 1.0;
        }

        // Compute the error and upate the best model if possible.
        Eigen::VectorXd new_error = coeff_t * model_better;

        double this_error = 0.0;
        for (const auto& inlier_idx : inlier_set) {
            this_error += std::abs(new_error(inlier_idx));
        }
        this_error /= inlier_set.size();

        if (inlier_set.size() > best_inlier_set.size()) {
            best_error = this_error;
            best_inlier_set = inlier_set;
        }
    }

    // Fill in the markers.
    inlier_markers.clear();
    inlier_markers.resize(pts1.size(), 0);
    for (const auto& inlier_idx : best_inlier_set) {
        inlier_markers[inlier_idx] = 1;
    }

    //printf("inlier ratio: %lu/%lu\n",
    //    best_inlier_set.size(), inlier_markers.size());
}

void ImageProcessor::featureUpdateCallback(
    std::deque<Feature_measure_t> &feature_buffer)
{
    // Publish features.
    std::vector<FeatureIDType> curr_ids(0);
    std::vector<cv::Point2f> curr_cam0_points(0);
    std::vector<cv::Point2f> curr_cam1_points(0);

    for (const auto& grid_features : (*curr_features_ptr)) {
        for (const auto& feature : grid_features.second) {
            curr_ids.push_back(feature.id);
            curr_cam0_points.push_back(feature.cam0_point);
            curr_cam1_points.push_back(feature.cam1_point);
        }
    }

    std::vector<cv::Point2f> curr_cam0_points_undistorted(0);
    std::vector<cv::Point2f> curr_cam1_points_undistorted(0);

    undistortPoints(
        curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
        cam0_distortion_coeffs, curr_cam0_points_undistorted);
    undistortPoints(
        curr_cam1_points, cam1_intrinsics, cam1_distortion_model,
        cam1_distortion_coeffs, curr_cam1_points_undistorted);

    Feature_measure_t feature_msg;
    feature_msg.features.resize(curr_ids.size());
    feature_msg.stamp = cam0_curr_img.stamp;
    for (uint32_t i = 0; i < curr_ids.size(); ++i) {
        feature_msg.features[i].id = (uint32_t)curr_ids[i];
        feature_msg.features[i].u0 = curr_cam0_points_undistorted[i].x;
        feature_msg.features[i].v0 = curr_cam0_points_undistorted[i].y;
        feature_msg.features[i].u1 = curr_cam1_points_undistorted[i].x;
        feature_msg.features[i].v1 = curr_cam1_points_undistorted[i].y;
    }
    feature_buffer.push_back(feature_msg);
}

void ImageProcessor::drawFeaturesMono()
{
    // Colors for different features.
    cv::Scalar tracked(0, 255, 0);
    cv::Scalar new_feature(0, 255, 255);

    static int32_t grid_height =
        cam0_curr_img.img.rows / processor_config.grid_row;
    static int32_t grid_width =
        cam0_curr_img.img.cols / processor_config.grid_col;

    // Create an output image.
    int32_t img_height = cam0_curr_img.img.rows;
    int32_t img_width = cam0_curr_img.img.cols;
    cv::Mat out_img(img_height, img_width, CV_8UC3);
    cvtColor(cam0_curr_img.img, out_img, CV_GRAY2RGB);

    // Draw grids on the image.
    for (int32_t i = 1; i < processor_config.grid_row; ++i) {
        cv::Point pt1(0, i * grid_height);
        cv::Point pt2(img_width, i * grid_height);
        cv::line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }
    for (int32_t i = 1; i < processor_config.grid_col; ++i) {
        cv::Point pt1(i * grid_width, 0);
        cv::Point pt2(i * grid_width, img_height);
        cv::line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    std::vector<FeatureIDType> prev_ids(0);
    for (const auto& grid_features : *prev_features_ptr) {
        for (const auto& feature : grid_features.second) {
            prev_ids.push_back(feature.id);
        }
    }

    // Collect feature points in the previous frame.
    std::map<FeatureIDType, cv::Point2f> prev_points;
    for (const auto& grid_features : *prev_features_ptr) {
        for (const auto& feature : grid_features.second) {
            prev_points[feature.id] = feature.cam0_point;
        }
    }

    // Collect feature points in the current frame.
    std::map<FeatureIDType, cv::Point2f> curr_points;
    for (const auto& grid_features : *curr_features_ptr) {
        for (const auto& feature : grid_features.second) {
            curr_points[feature.id] = feature.cam0_point;
        }
    }

    // Draw tracked features.
    for (const auto& id : prev_ids) {
        if (prev_points.find(id) != prev_points.end() &&
            curr_points.find(id) != curr_points.end()) {
            cv::Point2f prev_pt = prev_points[id];
            cv::Point2f curr_pt = curr_points[id];
            circle(out_img, curr_pt, 3, tracked);
            line(out_img, prev_pt, curr_pt, tracked, 1);

            prev_points.erase(id);
            curr_points.erase(id);
        }
    }

    // Draw new features.
    for (const auto& new_curr_point : curr_points) {
        cv::Point2f pt = new_curr_point.second;
        circle(out_img, pt, 3, new_feature, -1);
    }

    cv::imshow("Feature", out_img);
    cv::waitKey(1);
}

void ImageProcessor::drawFeaturesStereo()
{
    // Colors for different features.
    cv::Scalar tracked(0, 255, 0);
    cv::Scalar new_feature(0, 255, 255);

    static int32_t grid_height =
        cam0_curr_img.img.rows / processor_config.grid_row;
    static int32_t grid_width =
        cam0_curr_img.img.cols / processor_config.grid_col;

    // Create an output image.
    int32_t img_height = cam0_curr_img.img.rows;
    int32_t img_width = cam0_curr_img.img.cols;
    cv::Mat out_img(img_height, img_width * 2, CV_8UC3);
    cvtColor(cam0_curr_img.img,
                out_img.colRange(0, img_width), CV_GRAY2RGB);
    cvtColor(cam1_curr_img.img,
                out_img.colRange(img_width, img_width * 2), CV_GRAY2RGB);

    // Draw grids on the image.
    for (int32_t i = 1; i < processor_config.grid_row; ++i) {
        cv::Point pt1(0, i*grid_height);
        cv::Point pt2(img_width * 2, i * grid_height);
        line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }
    for (int32_t i = 1; i < processor_config.grid_col; ++i) {
        cv::Point pt1(i * grid_width, 0);
        cv::Point pt2(i * grid_width, img_height);
        line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }
    for (int32_t i = 1; i < processor_config.grid_col; ++i) {
        cv::Point pt1(i * grid_width + img_width, 0);
        cv::Point pt2(i * grid_width + img_width, img_height);
        line(out_img, pt1, pt2, cv::Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    std::vector<FeatureIDType> prev_ids(0);
    for (const auto& grid_features : *prev_features_ptr) {
        for (const auto& feature : grid_features.second) {
            prev_ids.push_back(feature.id);
        }
    }

    // Collect feature points in the previous frame.
    std::map<FeatureIDType, cv::Point2f> prev_cam0_points;
    std::map<FeatureIDType, cv::Point2f> prev_cam1_points;
    for (const auto& grid_features : *prev_features_ptr) {
        for (const auto& feature : grid_features.second) {
            prev_cam0_points[feature.id] = feature.cam0_point;
            prev_cam1_points[feature.id] = feature.cam1_point;
        }
    }

    // Collect feature points in the current frame.
    std::map<FeatureIDType, cv::Point2f> curr_cam0_points;
    std::map<FeatureIDType, cv::Point2f> curr_cam1_points;
    for (const auto& grid_features : *curr_features_ptr) {
        for (const auto& feature : grid_features.second) {
            curr_cam0_points[feature.id] = feature.cam0_point;
            curr_cam1_points[feature.id] = feature.cam1_point;
        }
    }

    // Draw tracked features.
    for (const auto& id : prev_ids) {
        if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
            curr_cam0_points.find(id) != curr_cam0_points.end()) {
            cv::Point2f prev_pt0 = prev_cam0_points[id];
            cv::Point2f prev_pt1 = prev_cam1_points[id]
                + cv::Point2f((float)img_width, 0.0f);
            cv::Point2f curr_pt0 = curr_cam0_points[id];
            cv::Point2f curr_pt1 = curr_cam1_points[id]
                + cv::Point2f((float)img_width, 0.0f);

            circle(out_img, curr_pt0, 3, tracked, -1);
            circle(out_img, curr_pt1, 3, tracked, -1);
            line(out_img, prev_pt0, curr_pt0, tracked, 1);
            line(out_img, prev_pt1, curr_pt1, tracked, 1);

            prev_cam0_points.erase(id);
            prev_cam1_points.erase(id);
            curr_cam0_points.erase(id);
            curr_cam1_points.erase(id);
        }
    }

    // Draw new features.
    for (const auto& new_cam0_point : curr_cam0_points) {
        cv::Point2f pt0 = new_cam0_point.second;
        cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
            cv::Point2f((float)img_width, 0.0f);

        circle(out_img, pt0, 3, new_feature, -1);
        circle(out_img, pt1, 3, new_feature, -1);
    }

    cv::imshow("Feature", out_img);
    cv::waitKey(1);
}

void ImageProcessor::updateFeatureLifetime()
{
    int32_t grid_cnt = processor_config.grid_row * processor_config.grid_col;
    for (int32_t code = 0; code < grid_cnt; ++code) {
        std::vector<FeatureMetaData>& features = (*curr_features_ptr)[code];
        for (const auto& feature : features) {
            if (feature_lifetime.find(feature.id) == feature_lifetime.end()) {
                feature_lifetime[feature.id] = 1;
            }
            else {
                ++feature_lifetime[feature.id];
            }
        }
    }
}

void ImageProcessor::featureLifetimeStatistics()
{
    std::map<int32_t, int32_t> lifetime_statistics;
    for (const auto& data : feature_lifetime) {
        if (lifetime_statistics.find(data.second) ==
            lifetime_statistics.end()) {
            lifetime_statistics[data.second] = 1;
        }
        else {
            ++lifetime_statistics[data.second];
        }
    }

    for (const auto& data : lifetime_statistics) {
        std::cout << data.first << " : " << data.second << std::endl;
    }
}

} // end namespace msckf_vio
