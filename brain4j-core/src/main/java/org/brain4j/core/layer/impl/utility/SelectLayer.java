package org.brain4j.core.layer.impl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class SelectLayer extends Layer {

    private final int index;

    public SelectLayer(int index) {
        this.index = index;
    }

    @Override
    public Layer connect(Layer previous) {
        return previous;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { inputs[index] };
    }

    @Override
    public int size() {
        return 0;
    }

    public int index() {
        return index;
    }
}
