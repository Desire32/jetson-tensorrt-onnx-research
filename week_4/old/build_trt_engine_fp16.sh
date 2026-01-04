#!/bin/bash
# ==========================================
# Build TensorRT engine from ONNX model
# Compatible with Jetson / TensorRT CLI
# ==========================================

# Fail on first error
set -e

# Paths
ONNX_MODEL="cypriotbert.onnx"
ENGINE_FILE="cypriotbert_fp16.engine"

# Check for ONNX file
if [ ! -f "$ONNX_MODEL" ]; then
    echo "❌ ONNX model not found: $ONNX_MODEL"
    exit 1
fi

# Run TensorRT conversion
echo "Building TensorRT engine..."
/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$ENGINE_FILE" \
    --fp16

echo "Engine saved to: $ENGINE_FILE"
