package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class MaxPoolLayer extends Layer {

    private int stride;
    private int windowHeight;
    private int windowWidth;
    private int size;

    private MaxPoolLayer() {
    }

    public MaxPoolLayer(int stride, int windowHeight, int windowWidth) {
        this.stride = stride;
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    @Override
    public Layer connect(Layer previous) {
        this.size = previous.size();
        return this;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { inputs[0].maxPoolGrad(stride, windowHeight, windowWidth) };
    }

    @Override
    public int size() {
        return size;
    }
}
