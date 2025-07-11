cmake_minimum_required(VERSION 3.10)
project(mediapipe_facemesh_pi)
set(CMAKE_CXX_STANDARD 17)

# --- TFLite Configuration ---
# Tell CMake where to find our custom FindTFLite.cmake module
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set these paths to match your system. These are the hints our find module needs.
set(TFLITE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tflite_build)
set(TENSORFLOW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow)

# Now, find the package using our module.
# CMake will look in CMAKE_MODULE_PATH and find "FindTFLite.cmake".
find_package(TFLite REQUIRED)
find_package(OpenCV REQUIRED)

# Define resource directories
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)

# --- Application ---
add_executable(mediapipe_facemesh_pi src/main.cpp)

# Include Json header
target_include_directories(mediapipe_facemesh_pi
                            PRIVATE
                            ${OpenCV_INCLUDE_DIRS}
                            third_party/json/include)

target_link_libraries(mediapipe_facemesh_pi
                        PRIVATE
                        TFLite::tflite
                        ${OpenCV_LIBS})

target_compile_definitions(mediapipe_facemesh_pi
                        PRIVATE
                        MODELS_DIR="${MODELS_DIR}"
)

file(COPY ${MODELS_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
