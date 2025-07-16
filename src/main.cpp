#include <opencv2/opencv.hpp>
#include "config_manager.h"
#include "face_manager.h"
#include "common_defines.h"
#include <json.hpp>
#include <chrono>

int main(int argc, char** argv) {

    // Load Configuration
    pi::FaceManagerConfig config;
    pi::ConfigManager::LoadFromFile(std::string(MODELS_DIR) + "/face_config.json", config);

    // Force V4L2 backend instead of GStreamer
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera with V4L2 backend." << std::endl;
        return -1;
    }

    std::cout << "Using backend: " << cap.getBackendName() << std::endl;

    // Check what the camera supports BEFORE setting properties
    std::cout << "=== Camera Capabilities ===" << std::endl;
    std::cout << "Default resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Default FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    // Setting properties in the correct order
    bool width_set = cap.set(cv::CAP_PROP_FRAME_WIDTH, config.frame_width);
    bool height_set = cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.frame_height);
    if (!width_set || !height_set) {
        std::cerr << "Error: Could not set camera resolution to " << config.frame_width << "x" << config.frame_height << std::endl;
        return -1;
    }
    bool fps_set = cap.set(cv::CAP_PROP_FPS, 30);
    bool buffer_set = cap.set(cv::CAP_PROP_BUFFERSIZE, 3);
    bool fourcc_set = cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));

    std::cout << "=== Property Setting Results ===" << std::endl;
    std::cout << "Width set: " << (width_set ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Height set: " << (height_set ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "FPS set: " << (fps_set ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Buffer set: " << (buffer_set ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "FOURCC set: " << (fourcc_set ? "SUCCESS" : "FAILED") << std::endl;

    // Get actual frame dimensions after setting
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "=== Final Camera Properties ===" << std::endl;
    std::cout << "Actual resolution: " << frame_width << "x" << frame_height << std::endl;
    std::cout << "Actual FPS: " << fps << std::endl;

    // Create a FaceDetection instance
    pi::FaceManager face_manager(config);

    // Pre-allocate matrices to avoid reallocation
    cv::Mat frame_bgr(frame_height, frame_width, CV_8UC3);
    cv::Mat frame_rgb(frame_height, frame_width, CV_8UC3);
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
    pi::DetectionBox detection_box;
    while (true) {
        // Read frame from camera
        if (!cap.read(frame_bgr)) {
            std::cerr << "Error: Could not read frame from camera." << std::endl;
            break;
        }

        if (frame_bgr.empty()) {
            std::cerr << "Error: Empty frame received from camera." << std::endl;
            continue;
        }

        // TICK - Start frame processing timer
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Convert frame to RGB
        cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

        // TICK - Start detection timer
        auto detection_start = std::chrono::high_resolution_clock::now();

        // Run face detection
        bool face_status = face_manager.Run(frame_rgb, detection_box);

        // TOCK - End detection timer
        auto detection_end = std::chrono::high_resolution_clock::now();
        auto detection_duration = std::chrono::duration<double, std::milli>(detection_end - detection_start);

        // TICK - Start drawing timer
        auto drawing_start = std::chrono::high_resolution_clock::now();

        if (face_status) {
            // Draw rotated detection box
            face_manager.draw_box_and_landmarks(frame_bgr, detection_box, face_manager.GetLandmarks());
        } else {
            cv::putText(frame_bgr, "No Face Detected", cv::Point(10, 30),
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
        cv::putText(frame_bgr, perf_text.str(), cv::Point(10, frame_height - 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // Display the frame
        cv::imshow("Face Detection", frame_bgr);

        frame_count++;

        // Handle keyboard input
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) { // 'q' or ESC key
            break;
        } else if (key == 's') { // 's' key to save current frame
            std::string filename = "face_detection_frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, frame_bgr);
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