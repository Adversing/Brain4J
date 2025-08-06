package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.List;

public class FlattenLayer extends Layer {

    private int dimension;

    public FlattenLayer(int dimension) {
        this.dimension = dimension;
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("dimension", value(dimension));
        return List.of();
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
