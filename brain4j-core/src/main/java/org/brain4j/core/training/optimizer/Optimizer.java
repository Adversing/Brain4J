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
public interface Optimizer {

    Tensor step(Tensor weights, Tensor gradient);

    default void initialize() {
        // Optional hook
    }
    
    default void postBatch() {
        // Optional hook
    }
}