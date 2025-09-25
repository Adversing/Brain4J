package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.index.Range;

public record SliceOperation(Range... ranges) implements Operation {

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].slice(ranges);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] inputShape = input.shape();

        Tensor gradInput = Tensors.zeros(inputShape);
        Range[] usedRanges = new Range[inputShape.length];

        for (int d = 0; d < inputShape.length; d++) {
            usedRanges[d] = (d < ranges.length) ? ranges[d] : null;
        }

        int[] expectedYShape = new int[inputShape.length];

        for (int d = 0; d < inputShape.length; d++) {
            int dim = inputShape[d];
            Range used = usedRanges[d];
            expectedYShape[d] = used != null ? used.size(dim) : dim;
        }

        gradOutput = gradOutput.reshape(expectedYShape);

        if (isContiguousSlice(inputShape, usedRanges)) {
            copyContiguous(gradInput, gradOutput, usedRanges);
        } else {
            int[] srcIndices = new int[inputShape.length];
            int[] dstIndices = new int[inputShape.length];
            sliceBackwardCopy(gradInput, gradOutput, usedRanges, srcIndices, dstIndices, 0);
        }

        return new Tensor[] { gradInput };
    }

    private boolean isContiguousSlice(int[] inputShape, Range[] ranges) {
        for (int d = 0; d < inputShape.length; d++) {
            Range r = ranges[d];
            if (r != null && r.step() != 1) return false;
        }
        return true;
    }

    private void copyContiguous(Tensor gradInput, Tensor gradOutput, Range[] ranges) {
        float[] gIn = gradInput.data();
        float[] gOut = gradOutput.data();

        int offset = 0;
        int length = gOut.length;

        int[] inShape = gradInput.shape();
        int lastDim = inShape.length - 1;
        Range range = ranges[lastDim];

        if (range != null) {
            int start = range.start(inShape[lastDim]);
            int end = range.end(inShape[lastDim]);

            if (!(start == 0 && end == inShape[lastDim])) {
                offset = range.start(inShape[lastDim]);
            }
        }

        for (int i = 0; i < length; i++) {
            gIn[offset + i] += gOut[i];
        }
    }

    private void sliceBackwardCopy(
        Tensor gradInput,
        Tensor gradOutput,
        Range[] ranges,
        int[] srcIndices,
        int[] dstIndices,
        int dim
    ) {
        int dims = srcIndices.length;
        int[] gradInputShape = gradInput.shape();

        if (dim == dims) {
            float vNum = gradOutput.get(dstIndices);
            float prevNum = gradInput.get(srcIndices);
            gradInput.set(prevNum + vNum, srcIndices);
            return;
        }

        Range range = ranges[dim];
        int start = 0;
        int end = gradInputShape[dim];
        int step = 1;

        if (range != null) {
            start = range.start(gradInputShape[dim]);
            end = range.end(gradInputShape[dim]);
            step = range.step();
        }

        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceBackwardCopy(gradInput, gradOutput, ranges, srcIndices, dstIndices, dim + 1);
        }
    }
}