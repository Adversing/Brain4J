package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.List;

public class InputLayer extends Layer {

    private int width;
    private int height;
    private int channels;

    public InputLayer(int width, int height, int channels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("width", value(width));
        layerBuilder.putAttrs("height", value(height));
        layerBuilder.putAttrs("channels", value(channels));
        return List.of();
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        return context.input().reshapeGrad(1, channels, height, width);
    }

    @Override
    public int size() {
        return channels;
    }
}
