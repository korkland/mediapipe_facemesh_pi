#include "face_detection.h"
#include <numeric>

namespace pi {

FaceDetection::FaceDetection(const FaceDetectionConfig& config)
    : TfLiteModel(config.model_path), config_(config) {

    // Initialize the FaceDetection with the provided configuration
    ASSERT(config_.frame_width > 0 && config_.frame_height > 0,
           "Invalid frame dimensions for FaceDetection.");

    // Build SSD anchors based on the configuration
    BuildAnchors(config_.anchor_config);

    // instead of computing 1 / (1+exp(-x)), we can use the sigmoid function directly
    config_.min_score_threshold = SigmoidInv(config_.min_score_threshold);

    // Verify the model input and output details
    ASSERT(GetInputTensorCount() == 1, "Expected exactly one input tensor for FaceDetection model.");
    ASSERT(GetInputTensorShape(0)->size == 4,
           "Expected input tensor shape to be [1, height, width, channels].");
    ASSERT(GetOutputTensorCount() == 2, "Expected exactly two output tensors for FaceDetection model.");

    // Allocate buffers based on model input dimensions
    auto input_rect_size = GetInputTensorShape(0)->data[1];
    auto input_rect_size_float = static_cast<float>(input_rect_size);
    tensor_scale_ = 1.0f / input_rect_size_float;
    model_input_ = cv::Mat(input_rect_size, input_rect_size, CV_32FC3, GetInputTensorData(0));
    model_input_ = -1.0f;
    resized_input_ = cv::Mat(input_rect_size, input_rect_size, CV_8UC3);

    // Set transform parameters for the model input
    const cv::RotatedRect rotated_rect(
        cv::Point2f(config_.frame_width / 2.0f, config_.frame_height / 2.0f),
        cv::Size2f(static_cast<float>(config_.frame_width), static_cast<float>(config_.frame_height)), 0.0f);

    cv::Mat source_points;
    cv::boxPoints(rotated_rect, source_points);
    std::array<float, 8> destination_corners = {
        0.0f,                   input_rect_size_float,
        0.0f,                   0.0f,
        input_rect_size_float,  0.0f,
        input_rect_size_float,  input_rect_size_float
    };
    cv::Mat destination_points(4, 2, CV_32F, destination_corners.data());
    model_input_perspective_transform_ = cv::getPerspectiveTransform(
        source_points, destination_points);

    const std::array<float, 6> projection_matrix{
        1.0f, 0.0f, 0.0f,
        0.0f, config_.frame_width / static_cast<float>(config_.frame_height),
        (-0.5f * config_.frame_width + config_.frame_height * 0.5f) * (1.0f / config_.frame_height)
    };
    std::copy(projection_matrix.begin(), projection_matrix.end(), projection_matrix_.begin());

    // Initialize SSD anchors
    ASSERT(anchors_.size() == GetOutputTensorShape(1)->data[1],
           "SSD anchors size does not match the output tensor size.");
    sorted_scores_indices_.resize(anchors_.size());
}

void FaceDetection::BuildAnchors(const AnchorConfig& config)
{
    int anchor_size = 0;
    for (const auto& stride: config.strides){
        auto feature_map_size = static_cast<int>(std::ceil(config.input_size / stride));
        anchor_size += (feature_map_size * feature_map_size * 2);
    }
    anchors_.clear();
    anchors_.reserve(anchor_size);

    int layer_id = 0;
    const int strides_size = config.strides.size();
    while (layer_id < strides_size){
        int last_same_stride_layer = layer_id;
        int aspect_ratio_size = 0;
        // For same strides, we merge the anchors in the same order.
        while (last_same_stride_layer < strides_size &&
                config.strides[layer_id] == config.strides[last_same_stride_layer]){
            aspect_ratio_size += 2;
            last_same_stride_layer++;
        }

        const int stride = config.strides[layer_id];
        auto feature_map_size = static_cast<int>(std::ceil(config.input_size / stride));

        for (int y = 0; y < feature_map_size; ++y){
            for (int x = 0; x < feature_map_size; ++x){
                for (int anchor_id = 0; anchor_id < aspect_ratio_size; ++anchor_id){
                    const float x_center = (x + config.anchor_offset) * 1.0f / feature_map_size;
                    const float y_center = (y + config.anchor_offset) * 1.0f / feature_map_size;

                    Point2D new_anchor;
                    new_anchor.x = x_center;
                    new_anchor.y = y_center;

                    anchors_.push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
}

bool FaceDetection::Run(const cv::Mat& image_in, DetectionBox& detection_box_out) {
    // Preprocess the input image
    PreprocessInput(image_in);

    // Run the model inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        return false;
    }

    // Postprocess the output to get the detection box
    return PostprocessOutput(detection_box_out);
}

void FaceDetection::PreprocessInput(const cv::Mat& image_in) {
    // Apply perspective transform
    cv::warpPerspective(image_in, resized_input_, model_input_perspective_transform_, model_input_.size());

    // Convert the image to float and normalize
    constexpr double scale_range_minus_one = 2.0f / 255.0f; // Scale factor to convert [0, 255] to [-1, 1]
    resized_input_.convertTo(model_input_, CV_32FC3, scale_range_minus_one, -1.0f);
}

bool FaceDetection::PostprocessOutput(DetectionBox& detection_box_out) {
    // Get the raw output tensor data
    float* output_boxes = GetOutputTensorData(0);
    float* output_scores = GetOutputTensorData(1);

    const int num_boxes = TensorsToDetections(output_boxes, output_scores);
    if (num_boxes == 0) {
        return false; // No detections found
    }

    // Perform non-max suppression to filter overlapping boxes
    WeightedNonMaxSuppression(output_boxes, output_scores, num_boxes);

    // Project the boxes to the original image coordinates
    DetectionProjection(output_boxes, detection_box_out);

    return true;
}

int FaceDetection::TensorsToDetections(float* boxes, float* scores) const {
    constexpr auto score_clipping_threshold = 100.0f; // Clipping threshold for scores

    const auto tensor_scale = tensor_scale_;
    const auto score_threshold = config_.min_score_threshold;

    int num_boxes_passed = 0;
    const int num_boxes = anchors_.size();
    for (int i = 0; i < num_boxes; ++i) {
        auto score = scores[i];
        if (score < score_threshold)
            continue; // Skip boxes with low scores

        const auto anchor = anchors_[i];
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
        score = Sigmoid(score);
        scores[num_boxes_passed] = score;

        num_boxes_passed++;
    }
    return num_boxes_passed;
}

void FaceDetection::WeightedNonMaxSuppression(float* boxes, float* scores, int num_boxes) {
    // Sort the scores and keep track of the indices
    std::iota(sorted_scores_indices_.begin(), sorted_scores_indices_.begin() + num_boxes, 0);
    std::sort(sorted_scores_indices_.begin(), sorted_scores_indices_.begin() + num_boxes,
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Non-max suppression algorithm - Note we are looking for the best candidate
    int candidates = 1;
    for (int i = 1; i < num_boxes; i++){
        if (OverlapSimilarity(&boxes[sorted_scores_indices_[0] * FaceDetectionIdx::FACE_DETECTION_COUNT],
                                &boxes[sorted_scores_indices_[i] * FaceDetectionIdx::FACE_DETECTION_COUNT])
                                > config_.min_suppression_threshold) {
            sorted_scores_indices_[candidates] = sorted_scores_indices_[i];
            candidates++;
        }
    }

    float total_score = 0.0f;
    std::array<float, FaceDetectionIdx::L_EYE_Y + 1> weighted_face{};
    for (int i = 0; i < candidates; i++) {
        const auto sorted_idx = sorted_scores_indices_[i];
        const auto candidate_score = scores[sorted_idx];
        const auto candidate_box = &boxes[sorted_idx * FaceDetectionIdx::FACE_DETECTION_COUNT];
        total_score += candidate_score;
        weighted_face[FaceDetectionIdx::X] += candidate_box[FaceDetectionIdx::X] * candidate_score;
        weighted_face[FaceDetectionIdx::Y] += candidate_box[FaceDetectionIdx::Y] * candidate_score;
        weighted_face[FaceDetectionIdx::WIDTH] += (candidate_box[FaceDetectionIdx::X] + candidate_box[FaceDetectionIdx::WIDTH]) * candidate_score;
        weighted_face[FaceDetectionIdx::HEIGHT] += (candidate_box[FaceDetectionIdx::Y] + candidate_box[FaceDetectionIdx::HEIGHT]) * candidate_score;
        weighted_face[FaceDetectionIdx::R_EYE_X] += candidate_box[FaceDetectionIdx::R_EYE_X] * candidate_score;
        weighted_face[FaceDetectionIdx::R_EYE_Y] += candidate_box[FaceDetectionIdx::R_EYE_Y] * candidate_score;
        weighted_face[FaceDetectionIdx::L_EYE_X] += candidate_box[FaceDetectionIdx::L_EYE_X] * candidate_score;
        weighted_face[FaceDetectionIdx::L_EYE_Y] += candidate_box[FaceDetectionIdx::L_EYE_Y] * candidate_score;
    }

    const auto scaled_score = 1.F / total_score;
    boxes[FaceDetectionIdx::X] = weighted_face[FaceDetectionIdx::X] * scaled_score;
    boxes[FaceDetectionIdx::Y] = weighted_face[FaceDetectionIdx::Y] * scaled_score;
    boxes[FaceDetectionIdx::WIDTH] = weighted_face[FaceDetectionIdx::WIDTH] * scaled_score - boxes[FaceDetectionIdx::X];
    boxes[FaceDetectionIdx::HEIGHT] = weighted_face[FaceDetectionIdx::HEIGHT] * scaled_score - boxes[FaceDetectionIdx::Y];
    boxes[FaceDetectionIdx::R_EYE_X] = weighted_face[FaceDetectionIdx::R_EYE_X] * scaled_score;
    boxes[FaceDetectionIdx::R_EYE_Y] = weighted_face[FaceDetectionIdx::R_EYE_Y] * scaled_score;
    boxes[FaceDetectionIdx::L_EYE_X] = weighted_face[FaceDetectionIdx::L_EYE_X] * scaled_score;
    boxes[FaceDetectionIdx::L_EYE_Y] = weighted_face[FaceDetectionIdx::L_EYE_Y] * scaled_score;
}

void FaceDetection::DetectionProjection(float* boxes, DetectionBox& detection_box_out) const {
    // Project the detection box to the original image coordinates
    constexpr float box_scale_size = 1.5f; // Scale factor for the box size

    auto frame_width = static_cast<float>(config_.frame_width);
    auto frame_height = static_cast<float>(config_.frame_height);

    // Project Keypoints - we need the keypoints to calculate the box rotation
    const auto right_eye_coords = ProjectPoint({ boxes[FaceDetectionIdx::R_EYE_X], boxes[FaceDetectionIdx::R_EYE_Y] }, projection_matrix_);
    boxes[FaceDetectionIdx::R_EYE_X] = right_eye_coords.x * frame_width;
    boxes[FaceDetectionIdx::R_EYE_Y] = right_eye_coords.y * frame_height;
    const auto left_eye_coords = ProjectPoint({ boxes[FaceDetectionIdx::L_EYE_X], boxes[FaceDetectionIdx::L_EYE_Y] }, projection_matrix_);
    boxes[FaceDetectionIdx::L_EYE_X] = left_eye_coords.x * frame_width;
    boxes[FaceDetectionIdx::L_EYE_Y] = left_eye_coords.y * frame_height;

    // Project bounding box.
    const auto xmin = boxes[FaceDetectionIdx::X];
    const auto ymin = boxes[FaceDetectionIdx::Y];
    const auto xmax = boxes[FaceDetectionIdx::X] + boxes[FaceDetectionIdx::WIDTH];
    const auto ymax = boxes[FaceDetectionIdx::Y] + boxes[FaceDetectionIdx::HEIGHT];
    const std::array<Point2D, 4> box_coords{
        ProjectPoint({ xmin, ymin }, projection_matrix_),
        ProjectPoint({ xmax, ymin }, projection_matrix_),
        ProjectPoint({ xmin, ymax }, projection_matrix_),
        ProjectPoint({ xmax, ymax }, projection_matrix_) };

    auto left_top = box_coords[0];
    auto right_bottom = box_coords[0];
    for (int i = 1; i < 4; i++)
    {
        left_top.x = std::min(left_top.x, box_coords[i].x);
        left_top.y = std::min(left_top.y, box_coords[i].y);
        right_bottom.x = std::max(right_bottom.x, box_coords[i].x);
        right_bottom.y = std::max(right_bottom.y, box_coords[i].y);
    }

    const auto width = (right_bottom.x - left_top.x) * frame_width;
    const auto height = (right_bottom.y - left_top.y) * frame_height;
    const auto longest_dim = std::max(width, height) * box_scale_size;

    detection_box_out.center.x = left_top.x * frame_width + width * 0.5f;
    detection_box_out.center.y = left_top.y * frame_height + height * 0.5f;
    detection_box_out.width = longest_dim;
    detection_box_out.height = longest_dim;
    detection_box_out.rotation = CalcRotation(boxes[FaceDetectionIdx::R_EYE_X], boxes[FaceDetectionIdx::R_EYE_Y],
                                            boxes[FaceDetectionIdx::L_EYE_X], boxes[FaceDetectionIdx::L_EYE_Y]);
}

} // namespace pi