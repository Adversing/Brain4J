package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.index.Range;

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
//        Tensor gradInput = Tensors.zeros(inputs[0].shape());
//        Tensor sliced = gradInput.slice(ranges);
//
//        float[] slicedData = sliced.data();
//        float[] gradOutData = gradOutput.data();
//
//        System.arraycopy(gradOutData, 0, slicedData, 0, slicedData.length);
//
//        return new Tensor[] { sliced };
        
        Tensor gradInput = Tensors.zeros(inputs[0].shape());
        Tensor view = gradInput.slice(ranges);
        
        float[] viewData = view.data();
        float[] gradOutData = gradOutput.data();
        System.arraycopy(gradOutData, 0, viewData, 0, gradOutData.length);
        
        return new Tensor[]{ gradInput };
    }
}
