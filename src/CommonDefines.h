#pragma once

#define ASSERT(condition, message) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " + std::string(message)); \
    }

namespace pi {

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

    DetectionBox() : center({0.0f, 0.0f}), width(0.0f), height(0.0f), rotation(0.0f) {}
    DetectionBox(Point2D center, float width = 0.0f, float height = 0.0f,
                 float rotation = 0.0f)
        : center(center), width(width), height(height), rotation(rotation) {}
};

// Utility functions
inline Point2D projectPoint(const Point2D& point, const std::array<float, 6>& transform) {
    return { point.x * transform[0] + point.y * transform[1] + transform[2],
             point.x * transform[3] + point.y * transform[4] + transform[5] };
}
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float sigmoid_inv(float x) { return -1.0f * std::log(1.0f / x - 1.0f); }
inline float overlap_similarity(const float* box1, const float* box2) {
    // Calculate the intersection area
    float x1 = std::max(box1[FaceDetectionIdx::X], box2[FaceDetectionIdx::X]);
    float y1 = std::max(box1[FaceDetectionIdx::Y], box2[FaceDetectionIdx::Y]);
    float x2 = std::min(box1[FaceDetectionIdx::X] + box1[FaceDetectionIdx::WIDTH], box2[FaceDetectionIdx::X] + box2[FaceDetectionIdx::WIDTH]);
    float y2 = std::min(box1[FaceDetectionIdx::Y] + box1[FaceDetectionIdx::HEIGHT], box2[FaceDetectionIdx::Y] + box2[FaceDetectionIdx::HEIGHT]);

    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    if (intersection_area == 0.0f) return 0.0f;

    // Calculate the union area
    float box1_area = box1[FaceDetectionIdx::WIDTH] * box1[FaceDetectionIdx::HEIGHT];
    float box2_area = box2[FaceDetectionIdx::WIDTH] * box2[FaceDetectionIdx::HEIGHT];
    float union_area = box1_area + box2_area - intersection_area;

    return intersection_area / union_area;
}
inline float normalize_radians(float angle) {
    // Normalize the angle to the range [-pi, pi]
    constexpr double two_pi = 2.0 * M_PI;
    constexpr double two_pi_inv = 1.0 / two_pi;
    return angle - std::floor((angle + M_PI) * two_pi_inv) * two_pi;
}
inline float calc_rotation(float x0, float y0, float x1, float y1, float target_angle = 0.0f) {
    // Calculate the angle in radians and convert to degrees
    float angle = target_angle - std::atan2(y0 - y1, x1 - x0);
    return normalize_radians(angle);
}

} // namespace pi