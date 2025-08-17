package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class ConvolveOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        return Tensors.convolve(inputs[0], inputs[1]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        // TODO: Fix gradient calculation
        Tensor A = inputs[0];
        Tensor B = inputs[1];

        // dL/dA = dL/dC * flip(B)
        Tensor gradA = gradOutput.convolve(B.flip());

        // dL/dB = flip(A) * dL/dC
        Tensor gradB = A.flip().convolve(gradOutput);

        return new Tensor[] { gradA, gradB };
    }
}
