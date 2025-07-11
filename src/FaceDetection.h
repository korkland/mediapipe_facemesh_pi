#pragma once

#include "TFliteModel.h"
#include "CommonDefines.h"

namespace pi {

struct FaceDetectionConfig {
    std::string model_path;
    int frame_width{ 0 };   // Width of the input frame
    int frame_height{ 0 };  // Height of the input frame
    float min_score_threshold{ 0.5f }; // Minimum score threshold for detection
    float min_suppression_threshold{ 0.3f }; // Minimum suppression threshold for non-max suppression
    std::vector<Point2D> ssd_anchors; // SSD anchors parameters
};

class FaceDetection : public TFliteModel {
public:
    explicit FaceDetection(const FaceDetectionConfig& config);
    virtual ~FaceDetection() = default;
    bool run(const cv::Mat& imageIn /* RGB */, DetectionBox& detectionBoxOut);

private:

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
    FaceDetectionConfig m_config; // Configuration for face detection

};
} // namespace pi