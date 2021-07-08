/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_CAM_STATE_H
#define MSCKF_VIO_CAM_STATE_H

#include "imu_state.h"

#include <map>

#include <Eigen/Dense>

namespace msckf_vio {

template <typename T>
class CAMStateStatic
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Takes a vector from the cam0 frame to the cam1 frame.
    static Eigen::Isometry3d T_cam0_cam1;
};

/*
 * @brief CAMState Stored camera state in order to
 *    form measurement model.
 */
class CAMState : public CAMStateStatic<void>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // An unique identifier for the CAM state.
    StateIDType id;

    // Time when the state is recorded
    uint64_t time;

    // Orientation
    // Take a vector from the world frame to the camera frame.
    Eigen::Vector4d orientation;

    // Position of the camera frame in the world frame.
    Eigen::Vector3d position;

    // These two variables should have the same physical
    // interpretation with `orientation` and `position`.
    // There two variables are used to modify the measurement
    // Jacobian matrices to make the observability matrix
    // have proper null space.
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;

    CAMState(): id(0), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d(0, 0, 0)) {}

    CAMState(const StateIDType& new_id ): id(new_id), time(0),
        orientation(Eigen::Vector4d(0, 0, 0, 1)),
        position(Eigen::Vector3d::Zero()),
        orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
        position_null(Eigen::Vector3d::Zero()) {}
};

typedef std::map<StateIDType, CAMState, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const StateIDType, CAMState> > > CamStateServer;

// Static member variables in CAMState class.
template <typename T>
Eigen::Isometry3d CAMStateStatic<T>::T_cam0_cam1(
    Eigen::Isometry3d::Identity());

} // namespace msckf_vio

#endif // MSCKF_VIO_CAM_STATE_H
