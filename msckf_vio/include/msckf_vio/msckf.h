#ifndef MSCKF_H
#define MSCKF_H

#include "msckf_vio/types.h"
#include "msckf_vio/params.h"

#include "buddy_memory.h"

#include <deque>
#include <mutex>
#include <condition_variable>

namespace msckf_vio {

class MsckfSystem
{
public:
    MsckfSystem(Stereo_camera_config_t& stereo_param);

    ~MsckfSystem();

    MsckfSystem() = delete;

    MsckfSystem& operator=(const MsckfSystem&) = delete;

    MsckfSystem& operator()(const MsckfSystem&) = delete;

    bool isImageLoopBusy();

    bool isFusionLoopBusy();

    bool insertStereoCam(
        const uint8_t *p_cam0_img,
        const uint8_t *p_cam1_img,
        const int32_t width,
        const int32_t height,
        const uint64_t timestamp);

    bool insertImu(
        const std::vector<Sensor_imu_t>& imu);

    void imageHandleLoop();

    void fusionHandleLoop();

    void reset();

    void showResult();

    // for debug
    bool checkImuBufferInOrder();

private:
    // image buffer
    std::mutex mtx_img;
    std::condition_variable cv_img;
    std::deque<Gray_img_t> cam0_img_buffer;
    std::deque<Gray_img_t> cam1_img_buffer;
    buddy_memory img_memory;

    // imu buffer
    std::mutex mtx_erase_imu;
    std::mutex mtx_imu;
    std::condition_variable cv_imu;
    std::deque<Sensor_imu_t> imu_buffer;

    // feature frame buffer
    std::mutex mtx_features;
    std::condition_variable cv_features;
    std::deque<Feature_measure_t> feature_buffer;

    // show result
    std::mutex mtx_show;
    std::condition_variable cv_show;
};

}

#endif
