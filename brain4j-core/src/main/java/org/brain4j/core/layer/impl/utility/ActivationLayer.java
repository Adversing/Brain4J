package org.brain4j.core.layer.impl.utility;

import org.brain4j.math.Commons;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.List;

public class ActivationLayer extends Layer {

    private int dimension;
    
    public ActivationLayer() {
    }
    
    public ActivationLayer(Activations activation) {
        this.activation = activation.function();
    }

    public ActivationLayer(Activation activation) {
        this.activation = activation;
    }

    @Override
    public Layer connect(Layer previous) {
        this.dimension = previous.size();
        return this;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < result.length; i++) {
            result[i] = inputs[i].activateGrad(activation);
        }

        cache.rememberOutput(this, result);
        return result;
    }

    @Override
    public int size() {
        return dimension;
    }
}
