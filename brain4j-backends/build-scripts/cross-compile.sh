#!/bin/bash
KERNEL_NAME=$1
KERNEL_SRC="../kernels/$KERNEL_NAME.slang"
OUT_DIR="src/main/resources/kernels"

# Metal
slangc -target metal "$KERNEL_SRC" -o "../backend-metal/$OUT_DIR/$KERNEL_NAME.metallib"
echo "Compiled Metal"

# CUDA
slangc -target ptx "$KERNEL_SRC" -o "../backend-cuda/$OUT_DIR/$KERNEL_NAME.ptx"
echo "Compiled CUDA"