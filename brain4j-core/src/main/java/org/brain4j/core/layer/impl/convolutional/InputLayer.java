package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.training.StatesCache;

import java.util.List;

public class InputLayer extends Layer {

    private int width;
    private int height;
    private int channels;
    
    public InputLayer() {
    }
    
    public InputLayer(int channels, int height, int width) {
        this.channels = channels;
        this.height = height;
        this.width = width;
    }
    
    @Override
    public void deserialize(List<org.brain4j.core.importing.proto.ProtoModel.Tensor> tensors, org.brain4j.core.importing.proto.ProtoModel.Layer layer) {
        this.width = SerializeUtils.attribute(layer, "width", 0);
        this.height = SerializeUtils.attribute(layer, "height", 0);
        this.channels = SerializeUtils.attribute(layer, "channels", 0);
    }
    
    @Override
    public void serialize(org.brain4j.core.importing.proto.ProtoModel.Layer.Builder builder) {
        builder.putAttrs("width", SerializeUtils.value(width));
        builder.putAttrs("height", SerializeUtils.value(height));
        builder.putAttrs("channels", SerializeUtils.value(channels));
    }
    
    @Override
    public Tensor forward(StatesCache cache, Tensor input) {
        int[] inputShape = input.shape();
        
        if (!validateInput(input)) {
            throw new IllegalArgumentException("Expecting 4-dimensional tensor with shape [batch, channels, height, width]!");
        }
        
        return input.reshapeGrad(inputShape[0], channels, height, width);
    }

    @Override
    public int size() {
        return channels;
    }
    
    @Override
    public boolean validateInput(Tensor input) {
        return input.rank() == 4;
    }
}
