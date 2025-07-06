#!/bin/bash
# setup.sh - Setup script for Raspberry Pi to build TensorFlow Lite and dependencies
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Update package list
echo "Updating package list..."
sudo apt update

# Install required packages
echo "Installing build tools and libraries..."
sudo apt install -y build-essential gdb cmake git pkg-config
sudo apt install -y libopencv-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev

# Clone TensorFlow repo if not already present
cd third_party
if [ ! -d "tensorflow" ]; then
    echo "Cloning TensorFlow repository..."
    git clone https://github.com/tensorflow/tensorflow.git
fi
cd tensorflow

git fetch --all
git checkout v2.14.0
git submodule update --init --recursive

# Create build directory if not exists
if [ ! -d "tflite_build" ]; then
    mkdir tflite_build
fi
cd tflite_build

# Run CMake and build
cmake ../tensorflow/lite -DCMAKE_BUILD_TYPE=Release -DTFLITE_ENABLE_XNNPACK=ON
make -j$(nproc)

echo "Setup complete! TensorFlow Lite built successfully."
