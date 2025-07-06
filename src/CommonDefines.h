#pragma once

namespace pi {

enum FaceDetectionIdx
{
    R_EYE_X = 4,
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

struct DetectionBox{
    float x; // X coordinate of the top-left corner
    float y; // Y coordinate of the top-left corner
    float width; // Width of the box
    float height; // Height of the box
    float rotation; // Rotation angle in degrees (optional, can be 0.0f if not used)

    DetectionBox(float x = 0.0f, float y = 0.0f, float width = 0.0f, float height = 0.0f,
                 float rotation = 0.0f)
        : x(x), y(y), width(width), height(height), rotation(rotation) {}
}

} // namespace pi