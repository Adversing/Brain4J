#!/bin/bash
KERNEL_NAME=$1
KERNEL_SRC="../kernels/$KERNEL_NAME.slang"
OUT_DIR="src/main/resources/kernels"

slangc -target ptx "$KERNEL_SRC" -o "../backend-cuda/$OUT_DIR/$KERNEL_NAME.ptx"
echo "Compiled CUDA"