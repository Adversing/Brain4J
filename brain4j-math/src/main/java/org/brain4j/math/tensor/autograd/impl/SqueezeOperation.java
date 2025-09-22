package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class SqueezeOperation implements Operation {

    private int dim = Integer.MAX_VALUE;

    public SqueezeOperation() {
    }

    public SqueezeOperation(int dim) {
        this.dim = dim;
    }

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor result = inputs[0].clone();
        return dim == Integer.MAX_VALUE ? result.squeeze() : result.squeeze(dim);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { dim == Integer.MAX_VALUE
            ? gradOutput.reshape(inputs[0].shape())
            : gradOutput.unsqueeze(dim)
        };
    }
}
