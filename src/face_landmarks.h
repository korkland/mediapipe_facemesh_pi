#pragma once

#include "tflite_model.h"
#include "common_defines.h"
#include "config_manager.h"

namespace pi {

class FaceLandmarks : public TfLiteModel {
public:
    explicit FaceLandmarks(const FaceLandmarksConfig& config);
    virtual ~FaceLandmarks() = default;
    bool Run(const cv::Mat& image_in /* RGB */, DetectionBox& detection_box_in_out /* Inplace assignment */);

    // Get the detected landmarks after successful inference
    const std::vector<Point3D>& GetLandmarks() const { return landmarks_; }

private:
    void PreprocessInput(const cv::Mat& image_in, const DetectionBox& detection_box_in_out);
    bool PostprocessOutput(DetectionBox& detection_box_in_out);
    void TensorsToLandmarks(DetectionBox& detection_box_in_out);
    void LandmarksToDetection(DetectionBox& detection_box_in_out);

    float tensor_scale_{ 1.0f }; // Scale factor for input tensor
    cv::Mat model_input_; // Input matrix for the model
    cv::Mat resized_input_; // Resized input matrix for the model
    cv::Mat rotated_source_; // Rotated source matrix for the model
    cv::Mat rotated_destination_; // Rotated destination matrix for the model
    std::vector<Point3D> landmarks_; // Detected landmarks
    FaceLandmarksConfig config_; // Configuration for face landmarks detection
};
} // namespace pi