#include "ConfigManager.h"
#include "CommonDefines.h"
#include <fstream>
#include <filesystem>
#include <json.hpp>

namespace pi {

void ConfigManager::loadFromFile(const std::string& config_path, FaceManagerConfig& config) {
    ASSERT(std::filesystem::exists(config_path), "Configuration file does not exist: " + config_path);
    ASSERT(std::filesystem::is_regular_file(config_path), "Configuration path is not a file: " + config_path);
    std::ifstream file(config_path);
    ASSERT(file.is_open(), "Failed to open configuration file: " + config_path);

    nlohmann::json json;
    file >> json;

    // Parse detection config
    ASSERT(json.contains("face_detection"), "Missing 'face_detection' section in configuration file.");
    parseDetectionConfig(json["face_detection"], config.detection_config);

    // Parse landmarks config
    ASSERT(json.contains("face_landmarks"), "Missing 'face_landmarks' section in configuration file.");
    parseLandmarksConfig(json["face_landmarks"], config.landmarks_config);

    // Parse manager config
    ASSERT(json.contains("face_manager"), "Missing 'face_manager' section in configuration file.");
    const auto& manager_json = json["face_manager"];
    ASSERT(manager_json.contains("frame_width"), "Missing 'frame_width' in face_manager section.");
    ASSERT(manager_json.contains("frame_height"), "Missing 'frame_height' in face_manager section.");
    config.frame_width = manager_json["frame_width"].get<int>();
    config.frame_height = manager_json["frame_height"].get<int>();
    config.detection_config.frame_width = config.frame_width;
    config.detection_config.frame_height = config.frame_height;
    config.landmarks_config.frame_width = config.frame_width;
    config.landmarks_config.frame_height = config.frame_height;

    if(manager_json.contains("enable_landmarks")) {
        config.enable_landmarks = manager_json["enable_landmarks"].get<bool>();
    }
    if(manager_json.contains("enable_performance_stats")) {
        config.enable_performance_stats = manager_json["enable_performance_stats"].get<bool>();
    }
}

void ConfigManager::parseAnchorConfig(const nlohmann::json& json, AnchorConfig& config) {
    ASSERT(json.contains("min_scale"), "Missing 'min_scale' in anchor configuration.");
    ASSERT(json.contains("max_scale"), "Missing 'max_scale' in anchor configuration.");
    ASSERT(json.contains("input_size"), "Missing 'input_size' in anchor configuration.");
    ASSERT(json.contains("anchor_offset"), "Missing 'anchor_offset' in anchor configuration.");
    ASSERT(json.contains("strides"), "Missing 'strides' in anchor configuration.");

    config.min_scale = json["min_scale"].get<float>();
    config.max_scale = json["max_scale"].get<float>();
    config.input_size = json["input_size"].get<float>();
    config.anchor_offset = json["anchor_offset"].get<float>();
    config.strides = json["strides"].get<std::vector<int>>();
}

void ConfigManager::parseDetectionConfig(const nlohmann::json& json, FaceDetectionConfig& config) {
    ASSERT(json.contains("model_path"), "Missing 'model_path' in face detection configuration.");
    ASSERT(json.contains("min_score_threshold"), "Missing 'min_score_threshold' in face detection configuration.");
    ASSERT(json.contains("min_suppression_threshold"), "Missing 'min_suppression_threshold' in face detection configuration.");
    ASSERT(json.contains("anchor_config"), "Missing 'anchor_config' in face detection configuration.");

    config.model_path = json["model_path"].get<std::string>();
    config.min_score_threshold = json["min_score_threshold"].get<float>();
    config.min_suppression_threshold = json["min_suppression_threshold"].get<float>();

    parseAnchorConfig(json["anchor_config"], config.anchor_config);
}

void ConfigManager::parseLandmarksConfig(const nlohmann::json& json, FaceLandmarksConfig& config) {
    ASSERT(json.contains("model_path"), "Missing 'model_path' in face landmarks configuration.");
    ASSERT(json.contains("min_score_threshold"), "Missing 'min_score_threshold' in face landmarks configuration.");

    config.model_path = json["model_path"].get<std::string>();
    config.min_score_threshold = json["min_score_threshold"].get<float>();
}
} // namespace pi
