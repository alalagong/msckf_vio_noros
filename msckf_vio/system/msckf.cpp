#include "msckf_vio/msckf.h"

#include "image_processor.h"
#include "msckf_vio.h"

#ifdef USING_VIZ
#include <opencv2/viz/vizcore.hpp>
#endif

#include <memory>

namespace msckf_vio {

// for c++14, using std::make_unique may better
static std::unique_ptr<ImageProcessor>
image_processor_ptr =
    std::unique_ptr<ImageProcessor>(new ImageProcessor);

static std::unique_ptr<MsckfVio>
msckf_vio_ptr = std::unique_ptr<MsckfVio>(new MsckfVio);

MsckfSystem::MsckfSystem(
    Stereo_camera_config_t& stereo_param) :
    img_memory(8, 102400)
{
    image_processor_ptr->initialize(stereo_param);
    msckf_vio_ptr->initialize(stereo_param);
}

bool MsckfSystem::isImageLoopBusy()
{
    if (cam0_img_buffer.size() > 5 ||
        cam1_img_buffer.size() > 5) {
        return true;
    }
    return false;
}

bool MsckfSystem::isFusionLoopBusy()
{
    if (feature_buffer.size() > 5) {
        return true;
    }
    return false;
}

bool MsckfSystem::insertStereoCam(
    const uint8_t *p_cam0_img,
    const uint8_t *p_cam1_img,
    const int32_t width,
    const int32_t height,
    const uint64_t timestamp)
{
    Gray_img_t cam0;
    cam0.stamp = timestamp;
    cam0.cols = width;
    cam0.rows = height;
    cam0.ptr = (uint8_t *)img_memory.mymalloc(width * height);
    memcpy(cam0.ptr, p_cam0_img, width * height);
    cam0_img_buffer.push_back(cam0);

    Gray_img_t cam1;
    cam1.stamp = timestamp;
    cam1.cols = width;
    cam1.rows = height;
    cam1.ptr = (uint8_t *)img_memory.mymalloc(width * height);
    memcpy(cam1.ptr, p_cam1_img, width * height);
    cam1_img_buffer.push_back(cam1);

    cv_img.notify_all();

    return true;
}

bool MsckfSystem::insertImu(
    const std::vector<Sensor_imu_t>& imu)
{
    std::lock_guard<std::mutex> lk(mtx_erase_imu);
    for (uint32_t i = 0; i < imu.size(); i++) {
        imu_buffer.push_back(imu[i]);
    }
    cv_imu.notify_all();

    return true;
}

void MsckfSystem::imageHandleLoop()
{
    while (1) {
        std::unique_lock<std::mutex> lk(mtx_img);
        cv_img.wait(lk);

        while (cam0_img_buffer.size() > 0) {
            const Gray_img_t& cam0 = cam0_img_buffer.front();
            const Gray_img_t& cam1 = cam1_img_buffer.front();

            image_processor_ptr->stereoCallback(cam0, cam1, imu_buffer);

            image_processor_ptr->drawFeaturesStereo();

            img_memory.myfree(cam0.ptr);
            cam0_img_buffer.pop_front();

            img_memory.myfree(cam1.ptr);
            cam1_img_buffer.pop_front();

            if (!msckf_vio_ptr->needInitGravityAndBias()) {
                image_processor_ptr->featureUpdateCallback(feature_buffer);
                cv_features.notify_all();
            }
        }
    }
}

void MsckfSystem::fusionHandleLoop()
{
    while (1) {
        if (msckf_vio_ptr->needInitGravityAndBias()) {
            std::unique_lock<std::mutex> lk(mtx_imu);
            cv_imu.wait(lk);

            if (imu_buffer.size() < 200) {
                continue;
            }
            msckf_vio_ptr->initializeGravityAndBias(imu_buffer);
        }

        std::unique_lock<std::mutex> lk(mtx_features);
        cv_features.wait(lk);

        while (feature_buffer.size() > 0) {
            const Feature_measure_t& measure = feature_buffer.front();
            const uint64_t measure_stamp = measure.stamp;
            msckf_vio_ptr->featureCallback(measure, imu_buffer);
            feature_buffer.pop_front();

            if (imu_buffer.size() > 0) {
                std::lock_guard<std::mutex> lk(mtx_erase_imu);
                auto it = imu_buffer.begin();
                auto itend = imu_buffer.end();
                while (it != itend && it->stamp <= measure_stamp) {
                    it++;
                }
                imu_buffer.erase(imu_buffer.cbegin(), it);
            }
        }
        cv_show.notify_all();
    }
}

void MsckfSystem::reset()
{
    msckf_vio_ptr->reset();
}

void MsckfSystem::showResult()
{
#ifdef USING_VIZ
    cv::viz::Viz3d window;
    window.setWindowSize(cv::Size(640, 480));
    window.setWindowPosition(cv::Point(0, 0));
    window.setBackgroundColor(cv::viz::Color::white());

    Odometry_t odom;
    std::vector<Vector3f_t> feature_pcl;

    std::vector<cv::Affine3f> traj;
    std::vector<cv::Vec3f> pointcloud;

    cv::Vec3f rot_vec(0.0f, 0.0f, 0.0f);
    while (1) {
        std::unique_lock<std::mutex> lk(mtx_show);
        cv_show.wait(lk);

        msckf_vio_ptr->getFusionResult(odom, feature_pcl);
        traj.emplace_back(rot_vec,
            cv::Vec3f(
                odom.position.x,
                odom.position.y,
                odom.position.z));

        for (uint32_t i = 0; i < feature_pcl.size(); i++) {
            pointcloud.emplace_back(
                feature_pcl[i].x,
                feature_pcl[i].y,
                feature_pcl[i].z);
        }

        if (traj.size() != 0 && !window.wasStopped()) {
            cv::viz::WTrajectory trajWidget(
                traj,
                cv::viz::WTrajectory::PATH,
                1.0f,
                cv::viz::Color(0, 255, 0));
            window.showWidget("traj", trajWidget);
        }

        if (pointcloud.size() != 0 && !window.wasStopped()) {
            cv::viz::WCloud cloudWidget(
                pointcloud,
                cv::viz::Color(255, 0, 0));
            cloudWidget.setRenderingProperty(
                cv::viz::POINT_SIZE,
                2.0f);
            window.showWidget("pointcloud", cloudWidget);
        }

        if (!window.wasStopped()) {
            window.spinOnce(1);
        }
    }
#endif
}

bool MsckfSystem::checkImuBufferInOrder()
{
    uint64_t time_pre = 0;
    for (const auto &imu : imu_buffer) {
        if (imu.stamp <= time_pre) {
            return false;
        }
        time_pre = imu.stamp;
    }
    return true;
}

}
