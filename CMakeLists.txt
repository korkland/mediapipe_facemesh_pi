cmake_minimum_required(VERSION 3.10)
project(mediapipe_facemesh_pi)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Raspberry Pi Optimization Flags ---
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    # ARM64 (64-bit) optimizations for RPi
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a -mtune=cortex-a72")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -ffast-math")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=armv8-a -mtune=cortex-a72")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize -ffast-math")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    # ARM32 (32-bit) optimizations for RPi
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8-a -mtune=cortex-a72")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4 -ftree-vectorize -ffast-math")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=armv8-a -mtune=cortex-a72")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon-vfpv4 -ftree-vectorize -ffast-math")
else()
    # x86/x64 optimizations for development/testing
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native")
endif()

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
endif()

# --- TFLite Configuration ---
# Tell CMake where to find our custom FindTFLite.cmake module
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Define the TFLite build directory and TensorFlow root directory
set(TFLITE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tflite_build)
set(TENSORFLOW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow)

# Add the TFLite build directory to the CMake module path
find_package(TFLite REQUIRED)
find_package(OpenCV REQUIRED)

# Define resource directories
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/models)

# --- Application ---
add_executable(mediapipe_facemesh_pi
                src/main.cpp
                src/tflite_model.cpp
                src/face_detection.cpp
                src/config_manager.cpp
                src/face_landmarks.cpp
                src/face_manager.cpp
                )

# Include Json header
target_include_directories(mediapipe_facemesh_pi
                            PRIVATE
                            ${OpenCV_INCLUDE_DIRS}
                            third_party/json
                            src)

target_link_libraries(mediapipe_facemesh_pi
                        PRIVATE
                        TFLite::tflite
                        ${OpenCV_LIBS})

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(mediapipe_facemesh_pi PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(mediapipe_facemesh_pi
                        PRIVATE
                        MODELS_DIR="${MODELS_DIR}"
                        CV_ENABLE_UNROLLED=1
                        CV_ENABLE_INTRINSICS=1
)

file(COPY ${MODELS_DIR} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Debug Information ---
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: Enabled")
else()
    message(STATUS "OpenMP: Not found")
endif()
