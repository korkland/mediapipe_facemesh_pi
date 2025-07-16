#pragma once

#include "tflite_model.h"
#include "common_defines.h"
#include "config_manager.h"

namespace pi {

class FaceDetection : public TfLiteModel {
public:
    explicit FaceDetection(const FaceDetectionConfig& config);
    virtual ~FaceDetection() = default;
    bool Run(const cv::Mat& image_in /* RGB */, DetectionBox& detection_box_out);

private:

    void BuildAnchors(const AnchorConfig& config);
    void PreprocessInput(const cv::Mat& image_in);
    bool PostprocessOutput(DetectionBox& detection_box_out);
    int TensorsToDetections(float* boxes, float* scores) const;
    void WeightedNonMaxSuppression(float* boxes, float*scores, int num_boxes);
    void DetectionProjection(float* boxes, DetectionBox& detection_box_out) const;

    float tensor_scale_{ 1.0f }; // Scale factor for input tensor
    cv::Mat model_input_; // Input matrix for the model
    cv::Mat resized_input_; // Resized input matrix for the model
    cv::Mat model_input_perspective_transform_; // Perspective transform matrix for the model
    std::array<float, 6> projection_matrix_;
    std::vector<int> sorted_scores_indices_; // Sorted indices of scores for non-max suppression
    std::vector<Point2D> anchors_; // SSD anchors parameters
    FaceDetectionConfig config_; // Configuration for face detection
};
} // namespace pi