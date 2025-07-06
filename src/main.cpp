#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Check if the input image path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // Load the input image
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return 1;
    }

    // Display the image
    cv::imshow("Input Image", image);
    cv::waitKey(0); // Wait for a key press

    return 0;
}