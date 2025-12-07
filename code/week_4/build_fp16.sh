#!/bin/bash
# ==========================================
# Build TensorRT engine from ONNX model
# Compatible with Jetson / TensorRT CLI
# ==========================================

# Fail on first error
set -e

MODEL_NAME="SmolLM2"
QUANT_TYPE="fp16"


ONNX_MODEL=$'{MODEL_NAME}.onnx'
ENGINE_FILE=$'{MODEL_NAME}_{QUANT_TYPE}.engine'

if [ ! -f "$ONNX_MODEL" ]; then
    echo "❌ ONNX model not found: $ONNX_MODEL"
    exit 1
fi

echo "Building TensorRT engine..."
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$ENGINE_FILE" \
    --fp16

echo "Engine saved to: $ENGINE_FILE"
