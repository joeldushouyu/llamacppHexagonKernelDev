#!/bin/bash

# Build script for GGML Hexagon test module
# This script builds a minimal test executable using ggml as a subdirectory

set -e  # Exit on error

echo "========================================="
echo "Building GGML Hexagon Test Module"
echo "========================================="

# Check required environment variables
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: ANDROID_NDK_ROOT environment variable is not set"
    exit 1
fi

if [ -z "$HEXAGON_SDK_ROOT" ]; then
    echo "ERROR: HEXAGON_SDK_ROOT environment variable is not set"
    exit 1
fi

# Try to find HEXAGON_TOOLS_ROOT if not set
if [ -z "$HEXAGON_TOOLS_ROOT" ]; then
    echo "HEXAGON_TOOLS_ROOT not set, searching for Hexagon tools..."
    # Common location in the Docker image
    if [ -d "$HEXAGON_SDK_ROOT/Tools/HEXAGON_Tools" ]; then
        # Find the latest version
        HEXAGON_TOOLS_ROOT=$(find "$HEXAGON_SDK_ROOT/Tools/HEXAGON_Tools" -maxdepth 1 -type d -name "*.*.*" | sort -V | tail -1)
        if [ -n "$HEXAGON_TOOLS_ROOT" ]; then
            export HEXAGON_TOOLS_ROOT
            echo "Found HEXAGON_TOOLS_ROOT: $HEXAGON_TOOLS_ROOT"
        else
            echo "WARNING: Could not find HEXAGON_TOOLS_ROOT automatically"
            echo "Setting to HEXAGON_SDK_ROOT as fallback"
            export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT"
        fi
    else
        echo "WARNING: Could not find Hexagon Tools directory"
        echo "Setting HEXAGON_TOOLS_ROOT to HEXAGON_SDK_ROOT as fallback"
        export HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT"
    fi
fi

echo "ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "HEXAGON_SDK_ROOT: $HEXAGON_SDK_ROOT"
echo "HEXAGON_TOOLS_ROOT: $HEXAGON_TOOLS_ROOT"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo ""
    echo "Creating build directory..."
    mkdir -p build
fi

# Configure the test module
echo ""
echo "Configuring test module..."
cmake \
    -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-31 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16 -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -B build \
    -S .

# Build the test module
echo ""
echo "Building test module..."
cmake --build build -j$(nproc)

# Create package directory
echo ""
echo "Creating package..."
PKG_DIR="pkg-test"
mkdir -p $PKG_DIR/bin
mkdir -p $PKG_DIR/lib

# Copy the test executable
cp build/bin/hexagon_test $PKG_DIR/bin/

# Copy required libraries
echo "Copying required libraries..."
cp build/ggml/src/libggml.so $PKG_DIR/lib/
cp build/ggml/src/libggml-base.so $PKG_DIR/lib/
cp build/ggml/src/libggml-cpu.so $PKG_DIR/lib/
cp build/ggml/src/ggml-hexagon/libggml-hexagon.so $PKG_DIR/lib/
cp build/ggml/src/ggml-opencl/libggml-opencl.so $PKG_DIR/lib/

# Copy Hexagon DSP libraries (skels)
echo "Copying Hexagon DSP libraries..."
if ls build/ggml/src/ggml-hexagon/libggml-htp-*.so 1> /dev/null 2>&1; then
    cp build/ggml/src/ggml-hexagon/libggml-htp-*.so $PKG_DIR/lib/
else
    echo "Warning: No Hexagon HTP libraries found"
fi

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo "Test executable: $PKG_DIR/bin/hexagon_test"
echo "Libraries: $PKG_DIR/lib/"
echo ""
echo "Run './deploy_and_test.sh' to push to Android device and test"
