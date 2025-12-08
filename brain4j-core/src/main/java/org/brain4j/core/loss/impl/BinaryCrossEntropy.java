package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class BinaryCrossEntropy implements LossFunction {

    private Tensor classWeights;

    public BinaryCrossEntropy() {
    }

    public BinaryCrossEntropy(Tensor classWeights) {
        this.classWeights = classWeights;
    }

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;
        int numClasses = actual.shape(actual.rank() - 1);

        for (int i = 0; i < actual.elements(); i++) {
            double y = actual.get(i);
            double p = predicted.get(i);
            double w = 1.0;

            if (classWeights != null) {
                int cls = i % numClasses;
                w = classWeights.get(cls);
            }

            loss -= w * (y * Math.log(p + 1e-15) + (1 - y) * Math.log(1 - p + 1e-15));
        }

        return loss / actual.shape(0);
    }

    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        if (classWeights == null) return error;

        Tensor weighted = error.clone();
        int numClasses = classWeights.shape(0);
        float[] w = classWeights.data();
        float[] data = weighted.data();

        for (int i = 0; i < data.length; i++) {
            int cls = i % numClasses;
            data[i] *= w[cls];
        }

        return weighted;
    }

    @Override
    public boolean isRegression() {
        return false;
    }

    public BinaryCrossEntropy classWeights(Tensor weights) {
        this.classWeights = weights;
        return this;
    }

    public Tensor classWeights() {
        return classWeights;
    }
}
