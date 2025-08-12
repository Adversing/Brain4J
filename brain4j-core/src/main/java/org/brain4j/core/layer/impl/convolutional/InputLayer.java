package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.List;

public class InputLayer extends Layer {

    private int width;
    private int height;
    private int channels;
    
    public InputLayer() {
    }
    
    public InputLayer(int width, int height, int channels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.width = SerializeUtils.attribute(layer, "width", 0);
        this.height = SerializeUtils.attribute(layer, "height", 0);
        this.channels = SerializeUtils.attribute(layer, "channels", 0);
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("width", SerializeUtils.value(width));
        builder.putAttrs("height", SerializeUtils.value(height));
        builder.putAttrs("channels", SerializeUtils.value(channels));
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
