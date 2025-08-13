package org.brain4j.core.loss.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.loss.LossFunction;

public class CrossEntropy implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss -= actual.get(i) * Math.log(predicted.get(i) + 1e-15);
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor delta(Tensor error, Tensor derivative) {
        return error;
    }

    @Override
    public boolean isRegression() {
        return false;
    }
}