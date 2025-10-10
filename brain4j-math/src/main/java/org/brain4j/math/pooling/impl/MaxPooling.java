package org.brain4j.math.pooling.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.pooling.PoolingProvider;
import org.brain4j.math.tensor.Tensor;

public class MaxPooling extends PoolingProvider {

    private int[] xCoords;
    private int[] yCoords;
    private int outHeight;
    private int outWidth;

    public MaxPooling(int stride, int windowHeight, int windowWidth) {
        super(stride, windowHeight, windowWidth);
    }

    @Override
    public Tensor pool(Tensor input) {
        int[] shape = input.shape();
        int rank = input.rank();

        if (rank < 2) {
            throw new IllegalArgumentException("Pooling requires at least 2D tensor.");
        }

        int inHeight = shape[rank - 2];
        int inWidth = shape[rank - 1];
        this.outHeight = (inHeight - windowHeight) / stride + 1;
        this.outWidth = (inWidth - windowWidth) / stride + 1;

        int outerSize = 1;
        for (int i = 0; i < rank - 2; i++) outerSize *= shape[i];

        float[] inputData = input.data();
        float[] outputData = new float[outerSize * outHeight * outWidth];

        this.xCoords = new int[outerSize * outHeight * outWidth];
        this.yCoords = new int[outerSize * outHeight * outWidth];

        for (int outer = 0; outer < outerSize; outer++) {
            pool2D(inputData, outputData, outer * inHeight * inWidth, outer * outHeight * outWidth, inHeight, inWidth);
        }

        int[] outShape = new int[rank];
        System.arraycopy(shape, 0, outShape, 0, rank - 2);
        outShape[rank - 2] = outHeight;
        outShape[rank - 1] = outWidth;

        return Tensors.create(outShape, outputData);
    }

    private void pool2D(float[] in, float[] out, int offsetIn, int offsetOut, int inH, int inW) {
        for (int h = 0; h < outHeight; h++) {
            for (int w = 0; w < outWidth; w++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                int maxY = -1;
                int maxX = -1;

                int hStart = h * stride;
                int wStart = w * stride;

                for (int kh = 0; kh < windowHeight; kh++) {
                    for (int kw = 0; kw < windowWidth; kw++) {
                        int ih = hStart + kh;
                        int iw = wStart + kw;
                        float value = in[offsetIn + ih * inW + iw];

                        if (value > maxVal) {
                            maxVal = value;
                            maxY = ih;
                            maxX = iw;
                        }
                    }
                }

                int outIndex = offsetOut + h * outWidth + w;
                out[outIndex] = maxVal;
                yCoords[outIndex] = maxY;
                xCoords[outIndex] = maxX;
            }
        }
    }

    @Override
    public Tensor backward(Tensor gradOutput, Tensor input) {
        int[] shape = input.shape();
        int rank = shape.length;

        int inHeight = shape[rank - 2];
        int inWidth = shape[rank - 1];

        int outerSize = 1;
        for (int i = 0; i < rank - 2; i++) outerSize *= shape[i];

        float[] gradIn = new float[input.elements()];
        float[] gradOut = gradOutput.data();

        for (int outer = 0; outer < outerSize; outer++) {
            int offIn = outer * inHeight * inWidth;
            int offOut = outer * outHeight * outWidth;
            backward2D(gradIn, gradOut, offIn, offOut, inWidth);
        }

        return Tensors.create(shape, gradIn);
    }

    private void backward2D(float[] gradIn, float[] gradOut, int offIn, int offOut, int inW) {
        for (int i = 0; i < outHeight * outWidth; i++) {
            int y = yCoords[offOut + i];
            int x = xCoords[offOut + i];
            gradIn[offIn + y * inW + x] += gradOut[offOut + i];
        }
    }
}