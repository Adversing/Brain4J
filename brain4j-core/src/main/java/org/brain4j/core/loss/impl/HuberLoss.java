package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

public class HuberLoss implements LossFunction {

    private final double delta;

    public HuberLoss() {
        this(1.0);
    }

    public HuberLoss(double delta) {
        this.delta = delta;
    }

    @Override
    public double calculate(Tensor expected, Tensor predicted) {
        double loss = 0.0;

        float[] expectedData = expected.data();
        float[] predictedData = predicted.data();

        for (int i = 0; i < expectedData.length; i++) {
            double diff = expectedData[i] - predictedData[i];
            double abs = Math.abs(diff);

            loss += abs <= delta
                ? 0.5 * diff * diff
                : delta * (abs - 0.5 * delta);
        }

        return loss / expectedData.length;
    }

    @Override
    public Tensor delta(Tensor error, Tensor derivative) {
        float[] errorData = error.data();
        float[] grad = new float[errorData.length];

        for (int i = 0; i < errorData.length; i++) {
            double value = errorData[i];
            double abs = Math.abs(value);
            double gradHuber = abs <= delta ? value : delta * Math.signum(value);
            grad[i] = (float) gradHuber;
        }

        return Tensors.create(error.shape(), grad).mul(derivative);
    }

    @Override
    public boolean isRegression() {
        return true;
    }

    public double delta() {
        return delta;
    }
}
