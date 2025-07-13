#pragma once

#include "TfLiteModel.h"
#include "CommonDefines.h"
#include "ConfigManager.h"

namespace pi {

class FaceDetection : public TfLiteModel {
public:
    explicit FaceDetection(const FaceDetectionConfig& config);
    virtual ~FaceDetection() = default;
    bool run(const cv::Mat& imageIn /* RGB */, DetectionBox& detectionBoxOut);

private:

    void buildAnchors(const AnchorConfig& config);
    void preprocessInput(const cv::Mat& input);
    bool postprocessOutput(DetectionBox& detectionBoxOut);
    int tensorsToDetections(float* boxes, float* scores) const;
    void weightedNonMaxSuppression(float* boxes, float*scores, int num_boxes);
    void detectionProjection(float* boxes, DetectionBox& detectionBoxOut) const;

    float m_tensor_scale{ 1.0f }; // Scale factor for input tensor
    cv::Mat m_model_input; // Input matrix for the model
    cv::Mat m_resized_input; // Resized input matrix for the model
    cv::Mat m_model_input_perspective_transform; // Perspective transform matrix for the model
    std::array<float, 6> m_projection_matrix;
    std::vector<int> m_sorted_scores_indices; // Sorted indices of scores for non-max suppression
    std::vector<Point2D> m_anchors; // SSD anchors parameters
    FaceDetectionConfig m_config; // Configuration for face detection
};
} // namespace pi