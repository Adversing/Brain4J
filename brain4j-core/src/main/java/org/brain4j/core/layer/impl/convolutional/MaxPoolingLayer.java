package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.List;

public class MaxPoolingLayer extends Layer {

    private int kernelWidth;
    private int kernelHeight;
    private int stride;
    private int channels;

    public MaxPoolingLayer(int kernelWidth, int kernelHeight) {
        this(kernelWidth, kernelHeight, 1);
    }

    public MaxPoolingLayer(int kernelWidth, int kernelHeight, int stride) {
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
    }

    @Override
    public Layer connect(Layer previous) {
        this.channels = previous.size();
        return this;
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.kernelWidth = attribute(layer, "kernel_width", 0);
        this.kernelHeight = attribute(layer, "kernel_height", 0);
        this.stride = attribute(layer, "stride", 0);
        this.channels = attribute(layer, "channels", 0);
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("kernel_width", value(kernelWidth));
        layerBuilder.putAttrs("kernel_height", value(kernelHeight));
        layerBuilder.putAttrs("stride", value(stride));
        layerBuilder.putAttrs("channels", value(channels));
        return List.of();
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input(); // [batch_size, channels, height, width]
        int[] shape = input.shape();

        int batchSize = shape[0];
        int channels = shape[1];
        int inputHeight = shape[2];
        int inputWidth = shape[3];

        int outHeight = (inputHeight - kernelHeight) / stride + 1;
        int outWidth = (inputWidth - kernelWidth) / stride + 1;

        Tensor output = Tensors.zeros(batchSize, channels, outHeight, outWidth);

        for (int b = 0; b < batchSize; b++) {
            for (int c = 0; c < channels; c++) {
                for (int oh = 0; oh < outHeight; oh++) {
                    for (int ow = 0; ow < outWidth; ow++) {
                        float max = Float.NEGATIVE_INFINITY;

                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                float value = input.get(b, c, ih, iw);

                                if (value > max) {
                                    max = value;
                                }
                            }
                        }

                        output.set(max, b, c, oh, ow);
                    }
                }
            }
        }

        return output;
    }

    @Override
    public int size() {
        return channels;
    }
}
