#pragma once

namespace pi {

#define ASSERT(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " + std::string(message)); \
    }

enum FaceDetectionIdx
{
    X = 0,
    Y,
    WIDTH,
    HEIGHT,
    R_EYE_X,
    R_EYE_Y,
    L_EYE_X,
    L_EYE_Y,
    NOSE_X,
    NOSE_Y,
    MOUTH_X,
    MOUTH_Y,
    R_EAR_X,
    R_EAR_Y,
    L_EAR_X,
    L_EAR_Y,
    FACE_DETECTION_COUNT
};

struct Point2D {
    float x; // X coordinate
    float y; // Y coordinate

    Point2D(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};

struct DetectionBox{
    Point2D center; // Center of the box
    float width;    // Width of the box
    float height;   // Height of the box
    float rotation; // Rotation angle in degrees (optional, can be 0.0f if not used)

    DetectionBox(Point2D center, float width = 0.0f, float height = 0.0f,
                 float rotation = 0.0f)
        : center(center), width(width), height(height), rotation(rotation) {}
};

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float sigmoid_inv(float x) { return -1.0f * std::log(1.0f / x - 1.0f); }

} // namespace pi