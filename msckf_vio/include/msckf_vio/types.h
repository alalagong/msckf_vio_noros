#ifndef TYPES_H
#define TYPES_H

#include <memory>
#include <stdint.h>
#include <vector>

namespace msckf_vio {

typedef struct {
    float x;
    float y;
    float z;
} Vector3f_t;

typedef struct {
    float x;
    float y;
    float z;
    float w;
} Quaternion_t;

typedef struct {
    uint32_t id;
    float u0;
    float v0;
    float u1;
    float v1;
} Stereo_feature_t;

typedef struct {
    uint64_t stamp;
    Vector3f_t angular_velocity;
    Vector3f_t linear_acceleration;
} Sensor_imu_t;

typedef struct {
    uint64_t stamp;
    std::vector<Stereo_feature_t> features;
} Feature_measure_t;

typedef struct {
    uint64_t stamp;

    Vector3f_t position;
    Quaternion_t orientation;
    double pose_covariance[36];

    Vector3f_t linear;
    Vector3f_t angular;
    double twist_covariance[36];
} Odometry_t;

struct Gray_img_t {
    Gray_img_t() :
        stamp(0),
        rows(0),
        cols(0),
        ptr(0) {}

    bool empty() {
        return (ptr == 0);
    }

    void create(
        int32_t _rows,
        int32_t _cols,
        void *_ptr = NULL) {
        rows = _rows;
        cols = _cols;
        if (ptr == NULL) {
            ptr = (uint8_t *)malloc(rows * cols);
        }
        else {
            ptr = (uint8_t *)_ptr;
        }
    }

    void release() {
        free(ptr);
        ptr = NULL;
    }

    uint64_t stamp;
    int32_t rows;
    int32_t cols;
    uint8_t *ptr;
};

}

#endif
