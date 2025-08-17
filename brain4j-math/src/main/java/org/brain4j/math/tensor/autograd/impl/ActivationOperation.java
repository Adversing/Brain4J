package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class ActivationOperation implements Operation {

    private final Activation activation;

    public ActivationOperation(Activation activation) {
        this.activation = activation;
    }

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return activation.activate(inputs[0]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor derivative = activation.derivative(inputs[0]); // ∂activation/∂x
        Tensor gradInput = gradOutput.times(derivative); // Chain rule: dL/dx = dL/dy * dy/dx

        return new Tensor[] { gradInput };
    }
}
