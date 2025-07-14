#pragma once

#include <json.hpp>
#include <string>
#include <vector>
#include "CommonDefines.h"

namespace pi {

struct AnchorConfig {
    float min_scale; // Minimum scale for anchors
    float max_scale; // Maximum scale for anchors
    float input_size; // Size of the input tensor
    float anchor_offset; // Offset for anchor center
    std::vector<int> strides; // Strides for feature maps
};

struct FaceDetectionConfig {
    std::string model_path; // Path to the face detection model
    int frame_width{ 0 };   // Width of the input frame
    int frame_height{ 0 };  // Height of the input frame
    float min_score_threshold{ 0.5f }; // Minimum score threshold for detection
    float min_suppression_threshold{ 0.3f }; // Minimum suppression threshold for non-max suppression
    AnchorConfig anchor_config; // Configuration for anchors
};

struct FaceLandmarksConfig {
    std::string model_path; // Path to the face landmarks model
    int frame_width{ 0 };   // Width of the input frame
    int frame_height{ 0 };  // Height of the input frame
    float min_score_threshold{ 0.5f }; // Minimum score threshold for landmarks detection
    int right_eye_index_for_rotation{ 33 }; // Index of the right eye landmark for rotation calculation
    int left_eye_index_for_rotation{ 263 }; // Index of the left eye landmark for rotation calculation
};

struct FaceManagerConfig {
    FaceDetectionConfig detection_config;
    FaceLandmarksConfig landmarks_config;
    int frame_width{ 0 };   // Width of the input frame
    int frame_height{ 0 };  // Height of the input frame
    bool enable_landmarks = true; // Enable face landmarks detection
    bool enable_performance_stats = true; // Enable performance statistics
};

class ConfigManager {
public:
    static void loadFromFile(const std::string& config_path, FaceManagerConfig& config);

private:
    static void parseAnchorConfig(const nlohmann::json& json, AnchorConfig& config);
    static void parseDetectionConfig(const nlohmann::json& json, FaceDetectionConfig& config);
    static void parseLandmarksConfig(const nlohmann::json& json, FaceLandmarksConfig& config);
};

} // namespace pi
