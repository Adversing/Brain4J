package org.brain4j.core.training.optimizer.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.training.optimizer.Optimizer;

public class GradientDescent implements Optimizer {

    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        return gradient;
    }
}
