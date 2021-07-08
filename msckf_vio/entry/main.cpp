#include "msckf_vio/msckf.h"

#include <fstream>
#include <regex>
#include <thread>

#include <opencv2/opencv.hpp>

using namespace msckf_vio;

bool read_image_list(
    std::string file,
    std::vector<uint64_t>& timestamp,
    std::vector<std::string>& image_list);

bool read_imu_data(
    std::string file,
    std::vector<Sensor_imu_t>& image_list);

bool read_stereo_image(
    cv::Mat& img,
    const std::string& file);

bool get_imus(
    std::vector<Sensor_imu_t>& imu,
    uint64_t timestamp0,
    uint64_t timestamp1,
    std::vector<Sensor_imu_t>& imu_interval);

int main(int argc, char *argv[])
{
    std::string data_dir;
    if (argc == 1) {
        data_dir = std::string("C:/Users/Rui/Desktop/V1_01_easy");
    }
    else {
        data_dir = std::string(argv[1]);
    }

    std::string config_file;
    if (argc < 3) {
        config_file = std::string(
            PROJECT_DIR
            "/msckf_vio/entry/config/"
            "camchain-imucam-euroc.yaml");
    }
    else {
        config_file = std::string(argv[2]);
    }

    Stereo_camera_config_t stereo_param;
    loadCaliParameters(config_file, stereo_param);

    std::vector<Sensor_imu_t> imu_data;
    read_imu_data(
        data_dir + "/imu0/data.csv",
        imu_data);

    std::vector<std::string> image_list;
    std::vector<uint64_t> image_stamp;
    read_image_list(
        data_dir + "/cam0/data.csv",
        image_stamp,
        image_list);

    MsckfSystem *pSystem = new MsckfSystem(stereo_param);
    pSystem->reset();

    std::thread *pThreadFeature =
        new std::thread(&MsckfSystem::imageHandleLoop, pSystem);

    std::thread *pThreadFusion =
        new std::thread(&MsckfSystem::fusionHandleLoop, pSystem);

    std::thread *pThreadViewer =
        new std::thread(&MsckfSystem::showResult, pSystem);

    cv::Mat cam0, cam1;
    std::vector<Sensor_imu_t> imu_interval;

    for (uint32_t i = 1; i < image_list.size(); i++) {
        while (pSystem->isImageLoopBusy()
            || pSystem->isFusionLoopBusy());

        read_stereo_image(
            cam0,
            data_dir + "/cam0/data/" + image_list[i]);

        read_stereo_image(
            cam1,
            data_dir + "/cam1/data/" + image_list[i]);

        get_imus(
            imu_data,
            image_stamp[i-1],
            image_stamp[i],
            imu_interval);
        pSystem->insertImu(imu_interval);

        pSystem->insertStereoCam(
            cam0.data,
            cam1.data,
            stereo_param.cam1_resolution[0],
            stereo_param.cam1_resolution[1],
            image_stamp[i]);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    pThreadFeature->join();
    pThreadFusion->join();

    return 0;
}

bool read_image_list(
    std::string file,
    std::vector<uint64_t>& timestamp,
    std::vector<std::string>& image_list)
{
#if 0
    FILE *p_file = fopen(file.c_str(), "r");
    fseek(p_file, 0L, SEEK_END);
    uint32_t file_size = ftell(p_file);
    rewind(p_file);

    std::string file_data;
    file_data.resize(file_size + 1);

    fread(&file_data[0], file_size, 1, p_file);
    file_data[file_size] = '\0';

    std::regex img_list_regex("(\\d+\\.png)");
    std::sregex_iterator iter(
        file_data.begin(),
        file_data.end(),
        img_list_regex);
    std::sregex_iterator end;
    while (iter != end) {
        image_list.push_back((*iter)[0]);
        iter++;
    }
#endif

    std::ifstream filestream(file);
    if (!filestream.is_open()) {
        return false;
    }

    uint64_t stamp;
    std::string filename;

    std::string line;
    char temp;
    while (getline(filestream, line)) {
        if(line.find("#") != std::string::npos) {
            continue;
        }

        std::stringstream ss(line);
        ss >> stamp >> temp >> filename;

        if(0 != filename.compare(std::to_string(stamp) + ".png")) {
            std::cout << "Read image error in : "
                << line << std::endl;
            continue;
        }

        timestamp.push_back(stamp);
        image_list.push_back(filename);
    }
    filestream.close();

    return true;
}

bool read_imu_data(
    std::string file,
    std::vector<Sensor_imu_t>& imu_data)
{
    std::ifstream filestream(file);
    if (!filestream.is_open()) {
        std::cout
            << "can't imu data: "
            << file << std::endl;
        return false;
    }

    std::string line;
    Sensor_imu_t imu;
    /* skip first line */
    getline(filestream, line);
    double vel[3], acc[3];
    while (getline(filestream, line)) {
        sscanf(line.c_str(), "%lld,%lf,%lf,%lf,%lf,%lf,%lf\n",
            &imu.stamp, &vel[0], &vel[1], &vel[2],
            &acc[0], &acc[1], &acc[2]);
        imu.angular_velocity.x = (float)vel[0];
        imu.angular_velocity.y = (float)vel[1];
        imu.angular_velocity.z = (float)vel[2];
        imu.linear_acceleration.x = (float)acc[0];
        imu.linear_acceleration.y = (float)acc[1];
        imu.linear_acceleration.z = (float)acc[2];
        imu_data.push_back(imu);
    }
    filestream.close();

    return true;
}

bool read_stereo_image(
    cv::Mat& img,
    const std::string& file)
{
    img = cv::imread(file, CV_8UC1);
    return true;
}

/* (timestamp0, timestamp1] */
bool get_imus(
    std::vector<Sensor_imu_t>& imu,
    uint64_t timestamp0,
    uint64_t timestamp1,
    std::vector<Sensor_imu_t>& imu_interval)
{
    static int32_t pre_index = 0;

    imu_interval.clear();

    uint32_t start_index = 0;
    for (int32_t i = pre_index; i >= 0; i--) {
        if (imu[i].stamp <= timestamp0) {
            break;
        }
        start_index = i;
    }

    for (uint32_t i = start_index; i < imu.size(); i++) {
        if (imu[i].stamp <= timestamp0) {
            continue;
        }
        if (imu[i].stamp <= timestamp1) {
            imu_interval.push_back(imu[i]);
        }
        else {
            break;
        }
        pre_index = i;
    }
    return true;
}
