#!/bin/bash

# Deploy and test script for GGML Hexagon test module
# Pushes the test executable and libraries to Android device and runs tests

set -e

echo "========================================="
echo "Deploying GGML Hexagon Test to Android"
echo "========================================="

# Configuration
PKG_DIR="pkg-test"
DEVICE_DIR="/data/local/tmp/hexagon_test"

# Check if package exists
if [ ! -d "$PKG_DIR" ]; then
    echo "ERROR: Package directory not found. Please run ./build_test.sh first"
    exit 1
fi

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "ERROR: No Android device found. Please connect device and enable USB debugging"
    exit 1
fi

echo "Device detected:"
adb devices

# Clean up old installation
echo ""
echo "Cleaning up old installation..."
adb shell "rm -rf $DEVICE_DIR" 2>/dev/null || true

# Create directory on device
echo "Creating directory on device..."
adb shell "mkdir -p $DEVICE_DIR/lib"

# Push test executable
echo ""
echo "Pushing test executable..."
adb push $PKG_DIR/bin/hexagon_test $DEVICE_DIR/

# Push libraries
echo ""
echo "Pushing libraries..."
adb push $PKG_DIR/lib/ $DEVICE_DIR/

# Make executable
adb shell "chmod +x $DEVICE_DIR/hexagon_test"

# Enable FARF logging for DSP
# 0x1f enables all FARF levels (ERROR, HIGH, MEDIUM, LOW, ALWAYS)
# Create .farf files for all HTP library versions
echo "Enabling FARF logging for all HTP library versions..."
for HTP_LIB in $(ls $PKG_DIR/lib/libggml-htp-v*.so 2>/dev/null); do
    HTP_LIB_NAME=$(basename "$HTP_LIB")
    FARF_FILE="${HTP_LIB_NAME%.so}.farf"
    echo "  Creating $FARF_FILE"
    adb shell "echo 0x1f > $DEVICE_DIR/lib/$FARF_FILE"
done

# Also create hexagon_test.farf (FARF looks for executable name)
echo "  Creating hexagon_test.farf (for executable-based logging)"
adb shell "echo 0x1f > $DEVICE_DIR/lib/hexagon_test.farf"

echo ""
echo "========================================="
echo "Running Tests on Device"
echo "========================================="
echo ""

# Set environment variables and run test
# GGML_HEXAGON_NDEV: Number of devices (HTP cores) to use
# GGML_HEXAGON_VERBOSE: Verbose mode (set to 1 to enable)
# GGML_HEXAGON_PROFILE: Profile mode (set to 1 to enable)

# Read environment variables or use defaults
NDEV=${NDEV:-1}
V=${V:-1}
PROF=${PROF:-1}

echo "Configuration:"
echo "  NDEV=$NDEV (number of Hexagon devices)"
echo "  V=$V (verbose mode)"
echo "  PROF=$PROF (profile mode)"
echo ""

# Clear logcat
adb logcat -c

# Start logcat in background to capture DSP FARF logs
adb logcat -v threadtime > /tmp/hexagon_logcat.txt &
LOGCAT_PID=$!
sleep 1

echo "Running test..."
set +e # Allow test to fail without exiting script
adb shell "cd $DEVICE_DIR && \
    export LD_LIBRARY_PATH=$DEVICE_DIR/lib:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=$DEVICE_DIR/lib && \
    export GGML_HEXAGON_NDEV=$NDEV && \
    export GGML_HEXAGON_VERBOSE=$V && \
    export GGML_HEXAGON_PROFILE=$PROF && \
    ./hexagon_test /data/local/tmp/gguf/mmproj-F16.gguf"
EXIT_CODE=$?
set -e

# Stop logcat
kill $LOGCAT_PID 2>/dev/null
wait $LOGCAT_PID 2>/dev/null || true

echo ""
echo "=== DSP/FARF Logs (from logcat) ==="
# Display all FARF logs from DSP (adsprpc tag contains DSP-side logs)
if [ -f /tmp/hexagon_logcat.txt ]; then
    #grep  "adsprpc" /tmp/hexagon_logcat.txt || echo "No DSP logs found"
    cat /tmp/hexagon_logcat.txt | grep "adsprpc" || echo "No DSP logs found"
else
    echo "Logcat file not found"
fi
echo "==================================="

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Tests completed successfully! ✓"
else
    echo "Tests failed with exit code: $EXIT_CODE ✗"
fi
echo "========================================="

exit $EXIT_CODE
