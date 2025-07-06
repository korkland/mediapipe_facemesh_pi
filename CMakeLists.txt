cmake_minimum_required(VERSION 3.10)
project(mediapipe_facemesh_pi)
set(CMAKE_CXX_STANDARD 17)

# --- TFLite Configuration ---
# Tell CMake where to find our custom FindTFLite.cmake module
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set these paths to match your system. These are the hints our find module needs.
set(TFLITE_BUILD_DIR "~/tensorflow/tflite_build")
set(TENSORFLOW_ROOT "~/tensorflow")

# Now, find the package using our module.
# CMake will look in CMAKE_MODULE_PATH and find "FindTFLite.cmake".
find_package(TFLite REQUIRED)
find_package(OpenCV REQUIRED)

# --- Application ---
add_executable(test_tflite test_tflite.cpp)

target_link_libraries(test_tflite
                        PRIVATE
                        TFLite::tflite
                        OpenCV::OpenCV
                        )
