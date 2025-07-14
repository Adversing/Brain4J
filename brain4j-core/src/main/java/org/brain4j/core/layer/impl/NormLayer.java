package org.brain4j.core.layer.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

/**
 * Implementation of a layer normalization layer, it's used to normalize inputs and improve training.
 * @author xEcho1337
 */
public class NormLayer extends Layer {

    private final double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon.
     */
    public NormLayer() {
        this(1e-5);
    }

    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public NormLayer(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.ones(previous.size()).withGrad();
        this.bias = Tensors.zeros(previous.size()).withGrad();

        return this;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        return input.layerNorm(epsilon).mul(weights).add(bias);
    }

    @Override
    public int size() {
        return weights.elements();
    }

    @Override
    public boolean skipPropagate() {
        return true;
    }
}
