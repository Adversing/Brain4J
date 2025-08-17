package org.brain4j.core.layer.impl.utility;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.core.importing.proto.ProtoModel;
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
    public Tensor forward(StatesCache cache, Tensor input) {
        return input.sliceGrad(ranges);
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        throw new UnsupportedOperationException("Not supported yet.");
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
