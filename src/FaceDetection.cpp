#include "FaceDetection.h"

namespace pi {

FaceDetection::FaceDetection(const FaceDetectionConfig& config)
    : TFliteModel(config.model_path), m_config(config) {

    // Initialize the FaceDetection with the provided configuration
    ASSERT(m_config.frame_width > 0 && m_config.frame_height > 0,
           "Invalid frame dimensions for FaceDetection.");

    // instead of computing 1 / (1+exp(-x)), we can use the sigmoid function directly
    m_config.min_score_threshold = sigmoid_inv(m_config.min_score_threshold);

    // Verify the model input and output details
    ASSERT(GetInputTensorCount() == 1, "Expected exactly one input tensor for FaceDetection model.");
    ASSERT(GetInputTensorShape(0).size() == 4,
           "Expected input tensor shape to be [1, height, width, channels]."); // TODO
    ASSERT(GetOutputTensorCount() == 2, "Expected exactly two output tensors for FaceDetection model.");

    // Set the input tensor dimensions based on the configuration
    auto input_rect_size = GetInputTensorShape(0)[1];
    auto input_rect_size_float = static_cast<float>(input_rect_size);
    m_tensor_scale = 1.0f / input_rect_size_float;
    m_model_input = cv::Mat(input_rect_size, input_rect_size, CV_32FC3, GetInputTensorData(0));
    // m_model_input = -1.0f; // TODO check if needed
    m_resized_input = cv::Mat(input_rect_size, input_rect_size, CV_8UC3);

    // Set transform parameters for the model input
    const cv::RotatedRect rotated_rect(
        cv::Point2f(m_config.frame_width / 2.0f, m_config.frame_height / 2.0f),
        cv::Size2f(static_cast<float>(m_config.frame_width), static_cast<float>(m_config.frame_height)), 0.0f);

    cv::Mat source_points;
    cv::boxPoints(rotated_rect, source_points);
    std::array<float, 8> destination_corners = {
        0.0f,                   input_rect_size_float,
        0.0f,                   0.0f,
        input_rect_size_float,  0.0f,
        input_rect_size_float,  input_rect_size_float
    };
    cv::Mat destination_points(4, 2, CV_32F, destination_corners.data());
    m_model_input_perspective_transform = cv::getPerspectiveTransform(
        source_points, destination_points);

    const std::array<float, 6> projection_matrix{
        1.0f, 0.0f, 0.0f,
        0.0f, m_config.frame_width / static_cast<float>(m_config.frame_height),
        (-0.5f * m_config.frame_width + m_config.frame_height * 0.5f) * (1.0f / m_config.frame_height)
    };
    std::copy(projection_matrix.begin(), projection_matrix.end(), m_projection_matrix.begin());

    // Initialize SSD anchors
    ASSERT(m_config.ssd_anchors.size() == GetOutputTensorShape(1)[1],
           "SSD anchors size does not match the output tensor size.");
    m_sorted_scores_indices.resize(m_config.ssd_anchors.size());
}

bool FaceDetection::run(const cv::Mat& imageIn, DetectionBox& detectionBoxOut) {
    // Preprocess the input image
    preprocessInput(imageIn);

    // Run the model inference
    if (m_interpreter->Invoke() != kTfLiteOk) {
        return false;
    }

    // Postprocess the output to get the detection box
    return postprocessOutput(detectionBoxOut);
}

void FaceDetection::preprocessInput(const cv::Mat& input) {
    // Apply perspective transform
    cv::warpPerspective(input, m_resized_input, m_model_input_perspective_transform, m_model_input.size());

    // Convert the image to float and normalize
    constexpr double scale_range_minus_one = 2.F / 255.F;
    m_resized_input.convertTo(m_model_input, CV_32FC3, scale_range_minus_one, -1.0f);
}

bool FaceDetection::postprocessOutput(DetectionBox& detectionBoxOut) {
    // Get the raw output tensor data
    float* output_boxes = GetOutputTensorData(0);
    float* output_scores = GetOutputTensorData(1);

    const int num_boxes = tensorsToDetections(output_boxes, output_scores);
    if (num_boxes == 0) {
        return false; // No detections found
    }

    // Perform non-max suppression to filter overlapping boxes
    weightedNonMaxSuppression(output_boxes, output_scores, num_boxes);

    // Project the boxes to the original image coordinates
    detectionProjection(output_boxes, detectionBoxOut);

    return true;
}

int FaceDetection::tensorsToDetections(float* boxes, float* scores) const {
    constexpr auto score_clipping_threshold = 100.0f; // Clipping threshold for scores

    const auto tensor_scale = m_tensor_scale;
    const auto score_threshold = m_config.min_score_threshold;

    int num_boxes_passed = 0;
    const int num_boxes = m_config.ssd_anchors.size();
    for (int i = 0; i < num_boxes; ++i) {
        auto score = scores[i];
        if (score < score_threshold)
            continue; // Skip boxes with low scores

        const auto anchor = m_config.ssd_anchors[i];
        const auto box_read = &boxes[i * FaceDetectionIdx::FACE_DETECTION_COUNT];
        const auto x_center = box_read[FaceDetectionIdx::X] * tensor_scale + anchor.x;
        const auto y_center = box_read[FaceDetectionIdx::Y] * tensor_scale + anchor.y;
        const auto width = box_read[FaceDetectionIdx::WIDTH] * tensor_scale;
        const auto height = box_read[FaceDetectionIdx::HEIGHT] * tensor_scale;

        if (width <= 0 || height <= 0)
            continue; // Skip invalid boxes

        // Store the box inplace
        auto box_write = &boxes[num_boxes_passed * FaceDetectionIdx::FACE_DETECTION_COUNT];
        box_write[FaceDetectionIdx::X] = x_center - width / 2.0f;
        box_write[FaceDetectionIdx::Y] = y_center - height / 2.0f;
        box_write[FaceDetectionIdx::WIDTH] = width;
        box_write[FaceDetectionIdx::HEIGHT] = height;
        box_write[FaceDetectionIdx::R_EYE_X] = box_read[FaceDetectionIdx::R_EYE_X] * tensor_scale + anchor.x;
        box_write[FaceDetectionIdx::R_EYE_Y] = box_read[FaceDetectionIdx::R_EYE_Y] * tensor_scale + anchor.y;
        box_write[FaceDetectionIdx::L_EYE_X] = box_read[FaceDetectionIdx::L_EYE_X] * tensor_scale + anchor.x;
        box_write[FaceDetectionIdx::L_EYE_Y] = box_read[FaceDetectionIdx::L_EYE_Y] * tensor_scale + anchor.y;

        // Save the score for sorting - clipped
        score = score < -score_clipping_threshold ? -score_clipping_threshold : score;
        score = score > score_clipping_threshold ? score_clipping_threshold : score;
        score = sigmoid(score);
        scores[num_boxes_passed] = score;

        num_boxes_passed++;
    }
    return num_boxes_passed;
}

} // namespace pi