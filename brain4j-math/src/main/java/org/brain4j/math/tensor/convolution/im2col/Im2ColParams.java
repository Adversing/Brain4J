package org.brain4j.math.tensor.convolution.im2col;

public record Im2ColParams(
    float[] inputData,
    float[] resultData,
    int inputBaseOffset,
    int resultBaseOffset,
    int channelCount,
    int inputHeight,
    int inputWidth,
    int filterHeight,
    int filterWidth,
    int outputHeight,
    int outputWidth
) {}