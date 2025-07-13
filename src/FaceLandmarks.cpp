#include "FaceLandmarks.h"

namespace pi {

FaceLandmarks::FaceLandmarks(const FaceLandmarksConfig& config)
    : TfLiteModel(config.model_path), m_config(config) {

    // Initialize the FaceLandmarks with the provided configuration
    ASSERT(m_config.frame_width > 0 && m_config.frame_height > 0,
        "Invalid frame dimensions for FaceLandmarks.");

    // instead of computing 1 / (1+exp(-x)), we can use the sigmoid function directly
    m_config.min_score_threshold = sigmoid_inv(m_config.min_score_threshold);

    // Verify the model input and output details
    ASSERT(GetInputTensorCount() == 1, "Expected exactly one input tensor for FaceLandmarks model.");
    ASSERT(GetInputTensorShape(0)->size == 4,
           "Expected input tensor shape to be [1, height, width, channels].");
    ASSERT(GetOutputTensorCount() == 2, "Expected exactly two output tensors for FaceLandmarks model.");

    // Allocate buffers based on model input dimensions
    auto input_rect_size = GetInputTensorShape(0)->data[1];
    auto input_rect_size_float = static_cast<float>(input_rect_size);
    m_tensor_scale = 1.0f / input_rect_size_float;
    m_model_input = cv::Mat(input_rect_size, input_rect_size, CV_32FC3, GetInputTensorData(0));

    std::array<float, 8> destination_corners = {
        0.0f,                   input_rect_size_float,
        0.0f,                   0.0f,
        input_rect_size_float,  0.0f,
        input_rect_size_float,  input_rect_size_float
    };
    m_rotated_source = cv::Mat(4, 2, CV_32F);
    m_rotated_destination = cv::Mat(4, 2, CV_32F);
    std::copy(destination_corners.begin(), destination_corners.end(), m_rotated_destination.ptr<float>());

    m_resized_input = cv::Mat(input_rect_size, input_rect_size, CV_8UC3);
    m_landmarks.resize(GetOutputTensorShape(0)->data[3] / 3); // Assuming 3D landmarks
}

bool FaceLandmarks::run(const cv::Mat& imageIn, DetectionBox& detectionBoxInOut) {
    // Preprocess the input image
    preprocessInput(imageIn, detectionBoxInOut);

    // Run the inference
    if (m_interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    // Postprocess the output to extract landmarks and detection box
    return postprocessOutput(detectionBoxInOut);
}

void FaceLandmarks::preprocessInput(const cv::Mat& ImageIn, const DetectionBox& detectionBoxInOut){
    // Crop and resize the input image based on the detection box
    const cv::RotatedRect rotated_rect(
        cv::Point2f(detectionBoxInOut.center.x, detectionBoxInOut.center.y),
        cv::Size2f(detectionBoxInOut.width, detectionBoxInOut.height),
        detectionBoxInOut.rotation * RAD2DEG);
    cv::boxPoints(rotated_rect, m_rotated_source);
    cv::warpPerspective(ImageIn, m_resized_input, cv::getPerspectiveTransform(
        m_rotated_source, m_rotated_destination), m_model_input.size());

    // Convert the resized input to float and normalize
    constexpr double scale_range_zero = 1.F / 255.f; // Scale factor to convert [0, 255] to [0, 1]
    m_resized_input.convertTo(m_model_input, CV_32FC3, scale_range_zero, 0.0f);
}

bool FaceLandmarks::postprocessOutput(DetectionBox& detectionBoxInOut) {
    // Extract the output tensors
    const float raw_score = GetOutputTensorData(1)[0];

    if (raw_score < m_config.min_score_threshold) {
        return false; // Early exit if score is below threshold
    }

    // Convert tensors to landmarks
    tensorsToLandmarks(detectionBoxInOut);

    // Convert landmarks to detection box
    landmarksToDetection(detectionBoxInOut);

    return true; // Inference successful
}
} // namespace pi