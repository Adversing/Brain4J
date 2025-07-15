package org.brain4j.core.layer.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

/**
 * Implementation of a batch normalization layer, it's used to normalize inputs and improve training.
 * @author xEcho1337
 */
public class BatchNormLayer extends Layer {

    private final double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon of 1e-5.
     */
    public BatchNormLayer() {
        this(1e-5);
    }

    /**
     * Constructs a new instance of a batch normalization layer.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public BatchNormLayer(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int size() {
        return 0;
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = input.variance();
        double std = Math.sqrt(variance + epsilon);
        return input.minus(mean).div(std);
    }
}
