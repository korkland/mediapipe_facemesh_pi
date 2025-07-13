#include <opencv2/opencv.hpp>
#include "FaceDetection.h"
#include "CommonDefines.h"
#include <json.hpp>
#include <chrono>

bool build_anchors(std::vector<pi::Point2D>& anchors)
{
    /* read anchor configuration*/
    // RET_CHECK(cfg.isMember("ssd_anchors_params"), "Configuration file {} is corrupted or missing ssd_anchors_params data. ", cfgFile);
    // const Json::Value& anchorCfg = cfg["ssd_anchors_params"];

    // RET_CHECK(anchorCfg.isMember("min_scale"), "Configuration file {} is corrupted or missing min_scale data. ", cfgFile);
    const float minScale = 0.1484375f; //anchorCfg["min_scale"].asFloat();

    // RET_CHECK(anchorCfg.isMember("max_scale"), "Configuration file {} is corrupted or missing max_scale data. ", cfgFile);
    const float maxScale = 0.75f; //anchorCfg["max_scale"].asFloat();

    // RET_CHECK(anchorCfg.isMember("input_size"), "Configuration file {} is corrupted or missing input_size data. ", cfgFile);
    const float inputSize = 128.f; //anchorCfg["input_size"].asFloat();

    // RET_CHECK(anchorCfg.isMember("anchor_offset_x"), "Configuration file {} is corrupted or missing anchor_offset_x data. ", cfgFile);
    const float anchorOffsetX = 0.5f; //anchorCfg["anchor_offset_x"].asFloat();

    // RET_CHECK(anchorCfg.isMember("anchor_offset_y"), "Configuration file {} is corrupted or missing anchor_offset_y data. ", cfgFile);
    const float anchorOffsetY = 0.5f; //anchorCfg["anchor_offset_y"].asFloat();

    // RET_CHECK(anchorCfg.isMember("strides"), "Configuration file {} is corrupted or missing strides data. ", cfgFile);
    // RET_CHECK((anchorCfg["strides"].size() > 1), "Configuration file {}. strides must be greater than 1!", cfgFile);
    // std::vector<int> strides(anchorCfg["strides"].size());
    // for (unsigned int i = 0; i < anchorCfg["strides"].size(); i++)
    //     strides[i] = anchorCfg["strides"][i].asInt();
    std::vector<int> strides = { 8, 16, 16, 16 }; // Example strides

    int anchorSize = 0;
    for (int i = 0; i < strides.size(); i++)
    {
        const int featureMapSize = std::ceil(inputSize / strides[i]);
        anchorSize += (featureMapSize * featureMapSize * 2);
    }
    anchors.reserve(anchorSize);

    int layerId = 0;
    const int stridesSize = strides.size();
    while (layerId < stridesSize)
    {
        int lastSameStrideLayer = layerId;
        int aspectRatioSize = 0;
        // For same strides, we merge the anchors in the same order.
        while (lastSameStrideLayer < stridesSize && strides[layerId] == strides[lastSameStrideLayer])
        {
            aspectRatioSize += 2;
            lastSameStrideLayer++;
        }

        const int stride = strides[layerId];
        const int featureMapHeight = std::ceil(inputSize / stride);
        const int featureMapWidth = std::ceil(inputSize / stride);

        for (int y = 0; y < featureMapHeight; ++y)
        {
            for (int x = 0; x < featureMapWidth; ++x)
            {
                for (int anchorId = 0; anchorId < aspectRatioSize; ++anchorId)
                {
                    const float xCenter = (x + anchorOffsetX) * 1.0F / featureMapWidth;
                    const float yCenter = (y + anchorOffsetY) * 1.0F / featureMapHeight;

                    pi::Point2D new_anchor;
                    new_anchor.x = xCenter;
                    new_anchor.y = yCenter;

                    anchors.push_back(new_anchor);
                }
            }
        }
        layerId = lastSameStrideLayer;
    }

    return true;
}

// Helper function to draw rotated bounding box
void drawRotatedBox(cv::Mat& image, const pi::DetectionBox& box, const cv::Scalar& color, int thickness) {
    constexpr double RAD2DEG = 180.0 / CV_PI;

    cv::RotatedRect rotatedRectangle(cv::Point2f(box.center.x, box.center.y),
                                    cv::Size(box.width, box.height), box.rotation * RAD2DEG);

    cv::Point2f vertices[4];
    rotatedRectangle.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(image, vertices[i], vertices[(i + 1) % 4], color, thickness);
}

int main(int argc, char** argv) {

    // Open default camera (index 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Set camera properties (optional)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Get actual frame dimensions
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "Camera properties: " << frame_width << "x" << frame_height << ", FPS: " << fps << std::endl;

    // Create a FaceDetectionConfig configuration
    pi::FaceDetectionConfig config;
    config.model_path = "models/face_detection_short_range.tflite";
    config.frame_width = frame_width;
    config.frame_height = frame_height;
    build_anchors(config.ssd_anchors);

    // Create a FaceDetection instance
    pi::FaceDetection faceDetection(config);

    cv::Mat frameBGR;
    cv::Mat frameRGB;
    int frame_count = 0;

    // Performance tracking variables
    double total_processing_time = 0.0;
    double total_detection_time = 0.0;
    double total_drawing_time = 0.0;
    double min_frame_time = std::numeric_limits<double>::max();
    double max_frame_time = 0.0;

    std::cout << "Starting camera feed..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save current frame" << std::endl;
    std::cout << "Frame | Total(ms) | Detection(ms) | Drawing(ms) | FPS" << std::endl;
    std::cout << "------|-----------|---------------|-------------|----" << std::endl;

    while (true) {
        // Read frame from camera
        if (!cap.read(frameBGR)) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }

        if (frameBGR.empty()) {
            std::cerr << "Error: Empty frame received from camera." << std::endl;
            continue;
        }

        // TICK - Start frame processing timer
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Convert frame to RGB
        cv::cvtColor(frameBGR, frameRGB, cv::COLOR_BGR2RGB);

        // TICK - Start detection timer
        auto detection_start = std::chrono::high_resolution_clock::now();

        // Run face detection
        pi::DetectionBox detectionBox;
        bool face_detected = faceDetection.run(frameRGB, detectionBox);

        // TOCK - End detection timer
        auto detection_end = std::chrono::high_resolution_clock::now();
        auto detection_duration = std::chrono::duration<double, std::milli>(detection_end - detection_start);

        // TICK - Start drawing timer
        auto drawing_start = std::chrono::high_resolution_clock::now();

        if (face_detected) {
            // Draw rotated detection box
            drawRotatedBox(frameBGR, detectionBox, cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(frameBGR, "No Face Detected", cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        // TOCK - End drawing timer
        auto drawing_end = std::chrono::high_resolution_clock::now();
        auto drawing_duration = std::chrono::duration<double, std::milli>(drawing_end - drawing_start);

        // TOCK - End frame processing timer
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_duration = std::chrono::duration<double, std::milli>(frame_end - frame_start);

        // Update performance statistics
        double frame_time_ms = frame_duration.count();
        double detection_time_ms = detection_duration.count();
        double drawing_time_ms = drawing_duration.count();

        total_processing_time += frame_time_ms;
        total_detection_time += detection_time_ms;
        total_drawing_time += drawing_time_ms;

        min_frame_time = std::min(min_frame_time, frame_time_ms);
        max_frame_time = std::max(max_frame_time, frame_time_ms);

        // Calculate current FPS
        double current_fps = 1000.0 / frame_time_ms;

        // Add performance info to frame
        std::stringstream perf_text;
        perf_text << std::fixed << std::setprecision(1)
                  << "Frame: " << frame_time_ms << "ms | "
                  << "Detection: " << detection_time_ms << "ms | "
                  << "FPS: " << current_fps;
        cv::putText(frameBGR, perf_text.str(), cv::Point(10, frame_height - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // Display the frame
        cv::imshow("Face Detection", frameBGR);

        frame_count++;

        // Handle keyboard input
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC key
            break;
        } else if (key == 's') { // 's' key to save current frame
            std::string filename = "face_detection_frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, frameBGR);
            std::cout << "Frame saved as: " << filename << std::endl;
        }

        // Display progress every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count << " | "
                      << "Processing: " << std::fixed << std::setprecision(2) << frame_time_ms << "ms | "
                      << "Detection: " << detection_time_ms << "ms | "
                      << "FPS: " << std::fixed << std::setprecision(1) << current_fps << std::endl;
        }
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    // Print final performance statistics
    std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Average processing time: " << std::fixed << std::setprecision(2)
              << (total_processing_time / frame_count) << " ms/frame" << std::endl;
    std::cout << "Average detection time: " << std::fixed << std::setprecision(2)
              << (total_detection_time / frame_count) << " ms/frame" << std::endl;
    std::cout << "Average drawing time: " << std::fixed << std::setprecision(2)
              << (total_drawing_time / frame_count) << " ms/frame" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1)
              << (1000.0 * frame_count / total_processing_time) << std::endl;
    std::cout << "Min frame time: " << std::fixed << std::setprecision(2)
              << min_frame_time << " ms" << std::endl;
    std::cout << "Max frame time: " << std::fixed << std::setprecision(2)
              << max_frame_time << " ms" << std::endl;
    std::cout << "Detection percentage: " << std::fixed << std::setprecision(1)
              << (100.0 * total_detection_time / total_processing_time) << "%" << std::endl;
    std::cout << "Drawing percentage: " << std::fixed << std::setprecision(1)
              << (100.0 * total_drawing_time / total_processing_time) << "%" << std::endl;

    std::cout << "\nCamera feed ended." << std::endl;
    return 0;
}