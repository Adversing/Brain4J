package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.List;

public class FlattenLayer extends Layer {

    private int dimension;
    
    public FlattenLayer() {
    }
    
    public FlattenLayer(int dimension) {
        this.dimension = dimension;
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.dimension = attribute(layer, "dimension", 0);
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("dimension", value(dimension));
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        int batchSize = input.shape()[0];

        return context.input().reshape(batchSize, dimension);
    }

    @Override
    public int size() {
        return dimension;
    }
}
