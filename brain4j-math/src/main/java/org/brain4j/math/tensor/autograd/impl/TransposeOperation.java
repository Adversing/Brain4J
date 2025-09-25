package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class TransposeOperation implements Operation {

    private final int dim1;
    private final int dim2;

    public TransposeOperation(int dim1, int dim2) {
        this.dim1 = dim1;
        this.dim2 = dim2;
    }

    @Override
    public int requiredInputs() {
        return 1;
    }
    
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].transpose(dim1, dim2);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { gradOutput.transpose(dim1, dim2) };
    }
}
