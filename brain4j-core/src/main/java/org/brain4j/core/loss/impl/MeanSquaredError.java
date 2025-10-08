package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class MeanSquaredError implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        float[] actualData = actual.data();
        float[] predictedData = predicted.data();
        
        for (int i = 0; i < actual.elements(); i++) {
            loss += Math.pow(actualData[i] - predictedData[i], 2);
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor delta(Tensor error, Tensor derivative) {
        return error.mul(derivative);
    }

    @Override
    public boolean isRegression() {
        return true;
    }
}