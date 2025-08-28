package org.brain4j.core.layer.impl.graph;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

public class GraphConvLayer extends Layer {

    private final int size;

    public GraphConvLayer(int size) {
        this.size = size;
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(previous.size(), size).withGrad();
        return this;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[0];
    }

    @Override
    public int size() {
        return size;
    }
}
