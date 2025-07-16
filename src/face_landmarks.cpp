#include "face_landmarks.h"

namespace pi {

FaceLandmarks::FaceLandmarks(const FaceLandmarksConfig& config)
    : TfLiteModel(config.model_path), config_(config) {

    // Initialize the FaceLandmarks with the provided configuration
    ASSERT(config_.frame_width > 0 && config_.frame_height > 0,
        "Invalid frame dimensions for FaceLandmarks.");

    // instead of computing 1 / (1+exp(-x)), we can use the sigmoid function directly
    config_.min_score_threshold = SigmoidInv(config_.min_score_threshold);

    // Verify the model input and output details
    ASSERT(GetInputTensorCount() == 1, "Expected exactly one input tensor for FaceLandmarks model.");
    ASSERT(GetInputTensorShape(0)->size == 4,
           "Expected input tensor shape to be [1, height, width, channels].");
    ASSERT(GetOutputTensorCount() == 2, "Expected exactly two output tensors for FaceLandmarks model.");

    // Allocate buffers based on model input dimensions
    auto input_rect_size = GetInputTensorShape(0)->data[1];
    auto input_rect_size_float = static_cast<float>(input_rect_size);
    tensor_scale_ = 1.0f / input_rect_size_float;
    model_input_ = cv::Mat(input_rect_size, input_rect_size, CV_32FC3, GetInputTensorData(0));

    std::array<float, 8> destination_corners = {
        0.0f,                   input_rect_size_float,
        0.0f,                   0.0f,
        input_rect_size_float,  0.0f,
        input_rect_size_float,  input_rect_size_float
    };
    rotated_source_ = cv::Mat(4, 2, CV_32F);
    rotated_destination_ = cv::Mat(4, 2, CV_32F);
    std::copy(destination_corners.begin(), destination_corners.end(), rotated_destination_.ptr<float>());

    resized_input_ = cv::Mat(input_rect_size, input_rect_size, CV_8UC3);
    landmarks_.resize(GetOutputTensorShape(0)->data[3] / 3); // Assuming 3D landmarks
}

bool FaceLandmarks::Run(const cv::Mat& image_in, DetectionBox& detection_box_in_out) {
    // Preprocess the input image
    PreprocessInput(image_in, detection_box_in_out);

    // Run the inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        return false;
    }

    // Postprocess the output to extract landmarks and detection box
    return PostprocessOutput(detection_box_in_out);
}

void FaceLandmarks::PreprocessInput(const cv::Mat& image_in, const DetectionBox& detection_box_in_out){
    // Crop and resize the input image based on the detection box
    const cv::RotatedRect rotated_rect(
        cv::Point2f(detection_box_in_out.center.x, detection_box_in_out.center.y),
        cv::Size2f(detection_box_in_out.width, detection_box_in_out.height),
        detection_box_in_out.rotation * RAD2DEG);
    cv::boxPoints(rotated_rect, rotated_source_);
    cv::warpPerspective(image_in, resized_input_, cv::getPerspectiveTransform(
        rotated_source_, rotated_destination_), model_input_.size());

    // Convert the resized input to float and normalize
    constexpr double scale_range_zero = 1.F / 255.f; // Scale factor to convert [0, 255] to [0, 1]
    resized_input_.convertTo(model_input_, CV_32FC3, scale_range_zero, 0.0f);
}

bool FaceLandmarks::PostprocessOutput(DetectionBox& detection_box_in_out) {
    // Extract the output tensors
    const float raw_score = GetOutputTensorData(1)[0];

    if (raw_score < config_.min_score_threshold) {
        return false; // Early exit if score is below threshold
    }

    // Convert tensors to landmarks
    TensorsToLandmarks(detection_box_in_out);

    // Convert landmarks to detection box
    LandmarksToDetection(detection_box_in_out);

    return true; // Inference successful
}

void FaceLandmarks::TensorsToLandmarks(DetectionBox& detection_box_in_out) {
    // Extract and scale landmarks from the output tensor
    const auto raw_landmarks = reinterpret_cast<Point3D*>(GetOutputTensorData(0));
    const auto sin_angle = std::sin(detection_box_in_out.rotation);
    const auto cos_angle = std::cos(detection_box_in_out.rotation);
    const auto z_scale = detection_box_in_out.width * tensor_scale_;
    for (size_t i = 0; i < landmarks_.size(); ++i) {
        const auto x = raw_landmarks[i].x * tensor_scale_ - 0.5f;
        const auto y = raw_landmarks[i].y * tensor_scale_ - 0.5f;

        Point3D landmark;
        landmark.x = (x * cos_angle - y * sin_angle) * detection_box_in_out.width + detection_box_in_out.center.x;
        landmark.y = (x * sin_angle + y * cos_angle) * detection_box_in_out.height + detection_box_in_out.center.y;
        landmark.z = raw_landmarks[i].z * z_scale; // Scale Z coordinate
        landmarks_[i] = landmark;
    }
}

void FaceLandmarks::LandmarksToDetection(DetectionBox& detection_box_in_out) {
    constexpr float box_scale_size = 1.5f;

    // Extract detection box based on landmarks
    auto [x_min_it, x_max_it] = std::minmax_element(landmarks_.begin(), landmarks_.end(),
        [](const Point3D& a, const Point3D& b) { return a.x < b.x; });
    auto [y_min_it, y_max_it] = std::minmax_element(landmarks_.begin(), landmarks_.end(),
        [](const Point3D& a, const Point3D& b) { return a.y < b.y; });

    // Calculate the center and size of the detection box
    const auto width = x_max_it->x - x_min_it->x;
    const auto height = y_max_it->y - y_min_it->y;
    const auto longest_dim = std::max(width, height) * box_scale_size;

    detection_box_in_out.center.x = x_min_it->x + width * 0.5f;
    detection_box_in_out.center.y = y_min_it->y + height * 0.5f;
    detection_box_in_out.width = longest_dim;
    detection_box_in_out.height = longest_dim;
    detection_box_in_out.rotation = CalcRotation(
        landmarks_[config_.right_eye_index_for_rotation].x, landmarks_[config_.right_eye_index_for_rotation].y,
        landmarks_[config_.left_eye_index_for_rotation].x, landmarks_[config_.left_eye_index_for_rotation].y);
}

} // namespace pi