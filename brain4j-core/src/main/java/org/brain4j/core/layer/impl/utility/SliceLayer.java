package org.brain4j.core.layer.impl.utility;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.List;

public class SliceLayer extends Layer {
    
    private final Range[] ranges;
    private int size;
    
    public SliceLayer(Range... ranges) {
        this.ranges = ranges;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            result[i] = inputs[i].sliceGrad(ranges);
        }

        return result;
    }
    
    @Override
    public Layer connect(Layer previous) {
        this.size = previous.size();
        return this;
    }
    
    @Override
    public int size() {
        return size;
    }
}
