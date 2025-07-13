#pragma once

#include "TfLiteModel.h"
#include "CommonDefines.h"
#include "ConfigManager.h"

namespace pi {

class FaceLandmarks : public TfLiteModel {
public:
    explicit FaceLandmarks(const FaceLandmarksConfig& config);
    virtual ~FaceLandmarks() = default;
    bool run(const cv::Mat& imageIn /* RGB */, DetectionBox& detectionBoxInOut /* Inplace assignment */);

    // Get the detected landmarks after successful inference
    const std::vector<Point3D>& getLandmarks() const { return m_landmarks; }

private:
    void preprocessInput(const cv::Mat& imageIn, const DetectionBox& detectionBoxInOut);
    bool postprocessOutput(DetectionBox& detectionBoxInOut);
    void tensorsToLandmarks(DetectionBox& detectionBoxInOut);
    void landmarksToDetection(DetectionBox& detectionBoxInOut);

    float m_tensor_scale{ 1.0f }; // Scale factor for input tensor
    cv::Mat m_model_input; // Input matrix for the model
    cv::Mat m_resized_input; // Resized input matrix for the model
    cv::Mat m_rotated_source; // Rotated source matrix for the model
    cv::Mat m_rotated_destination; // Rotated destination matrix for the model
    std::vector<Point3D> m_landmarks; // Detected landmarks
    FaceLandmarksConfig m_config; // Configuration for face landmarks detection
};
} // namespace pi