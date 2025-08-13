package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class TransposeOperation implements Operation {
    
    @Override
    public int requiredInputs() {
        return 1;
    }
    
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].transpose();
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { gradOutput.transpose() };
    }
}
