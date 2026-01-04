#!/bin/bash
# ==========================================
# Benchmark TensorRT engine performance
# ==========================================

ENGINE=${1:-cypriotbert_fp16.engine}
SHAPE=${2:-input_ids:1x7}
echo "Benchmarking $ENGINE ..."
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=$ENGINE \
    --shapes=$SHAPE \
    --avgRuns=100 \
    --separateProfileRun \
    --verbose >> benchmark_fp16.txt
