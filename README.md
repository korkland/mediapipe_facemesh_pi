# Mediapipe FaceMesh C++ (Raspberry Pi)

Fun project to demonstrate Mediapipe Facemesh flow in C++.
Full implementation of MediaPipe FaceMesh in C++ using TensorFlow Lite (TFLite) and OpenCV, optimized for Raspberry Pi.

## Features
- Real-time face mesh detection using TFLite
- OpenCV for video capture and visualization
- Lightweight and optimized for ARM (Raspberry Pi 4)

## Requirements
- Raspberry Pi 4 (Raspberry Pi OS recommended)
- C++17 or newer
- CMake >= 3.10
- OpenCV (>= 4.x)
- TensorFlow Lite (ARM build)

## Setup

### 1. Run the Setup Script
```
bash setup.sh
```

This script will install all required dependencies, including OpenCV and TensorFlow Lite (if configured). Review and edit `setup.sh` as needed for your environment.

### 2. Clone the Repository
```
git clone https://github.com/yourusername/mediapipe_facemesh_pi.git
cd mediapipe_facemesh_pi
```

### 3. Build the Project
```
mkdir build
cd build
cmake ..
make -j4
```

### 4. Run the Demo
```
./facemesh_demo
```

## Usage
- The demo will open the default camera and display face mesh landmarks in real time.
- Press `q` to quit.

## Project Structure
- `CMakeLists.txt` - Build configuration
- `cmake/` - CMake modules (e.g., FindTFLite.cmake)
- `build/` - Build output
- `setup.sh` - Optional setup script
- `src/` - Source code (add your implementation here)

## References
- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [OpenCV](https://opencv.org/)

## License
See [LICENSE](LICENSE).
