package org.brain4j.core.layer.impl.utility;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.List;

public class SqueezeLayer extends Layer {
    
    private int dimension;
    private int size;
    
    public SqueezeLayer(int dimension) {
        this.dimension = dimension;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] results = new Tensor[inputs.length];

        for (int i = 0; i < results.length; i++) {
            Tensor input = inputs[i];
            results[i] = dimension == -1 ? input.squeeze() : input.squeeze(dimension);
        }

        return results;
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("dimension", SerializeUtils.value(dimension));
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.dimension = SerializeUtils.attribute(layer, "dimension", 0);
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
