package org.brain4j.core.training.optimizer;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;

/**
 * Abstract class to define a gradient optimizer.
 * @see GradientDescent
 * @see Adam
 * @see AdamW
 * @see Lion
 */
public abstract class Optimizer {

    protected double learningRate;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public abstract Tensor step(Tensor weights, Tensor gradient);

    public void initialize() {
        // No-op
    }

    public void postBatch() {
        // No-op
    }

    public double learningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}