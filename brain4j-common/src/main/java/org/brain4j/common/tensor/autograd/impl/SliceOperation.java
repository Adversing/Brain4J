package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.index.Range;

import java.util.Arrays;

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
            if (usedRanges[d] != null) {
                expectedYShape[d] = usedRanges[d].size(inputShape[d]);
            } else {
                expectedYShape[d] = inputShape[d];
            }
        }
        
        gradOutput = gradOutput.reshape(expectedYShape);
        
        int[] srcIndices = new int[inputShape.length];
        int[] dstIndices = new int[inputShape.length];
        
        sliceBackwardCopy(gradInput, gradOutput, usedRanges, srcIndices, dstIndices, 0);
        
        return new Tensor[] { gradInput };
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
