package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.index.Range;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ConvLayer extends Layer {

    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int channels;
    private int stride = 1;
    private int padding = 0;
    
    public ConvLayer() {
    }
    
    public ConvLayer(Activations activation, int channels, int filters, int kernelWidth, int kernelHeight) {
        this.activation = activation.function();
        this.channels = channels;
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
    }
    
    @Override
    public Layer connect(Layer previous) {
        this.bias = Tensors.zeros(filters).withGrad();
        this.weights = Tensors.zeros(filters, channels, kernelHeight, kernelWidth).withGrad();

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.bias.map(x -> weightInit.generate(generator, input, output));
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.filters = attribute(layer, "filters", 0);
        this.kernelWidth = attribute(layer, "kernel_width", 0);
        this.kernelHeight = attribute(layer, "kernel_height", 0);
        this.stride = attribute(layer, "stride", 0);
        this.padding = attribute(layer, "padding", 0);
        
        for (ProtoModel.Tensor tensor : tensors) {
            String name = tensor.getName().split("\\.")[2];
            switch (name) {
                case "weight" -> this.weights = deserializeTensor(tensor);
                case "bias" -> this.bias = deserializeTensor(tensor);
            }
        }
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("filters", value(filters));
        builder.putAttrs("kernel_width", value(kernelWidth));
        builder.putAttrs("kernel_height", value(kernelHeight));
        builder.putAttrs("stride", value(stride));
        builder.putAttrs("padding", value(padding));
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();

        if (!validateInput(input)) {
            throw new IllegalArgumentException("Input dimension mismatch! Got: " + Arrays.toString(input.shape()));
        }

        // [batch_size, channels, height, width]
        int batchSize = input.shape()[0];
        Tensor[] tensors = new Tensor[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Tensor batch = input.slice(Range.point(i));
            Tensor result = batch.convolveGrad(weights);

            tensors[i] = result;
        }

        return Tensors.concatGrad(List.of(tensors), 0);
    }

    @Override
    public int size() {
        return filters;
    }

    @Override
    public boolean validateInput(Tensor input) {
        // [batch_size, channels, height, width]
        return input.rank() == 4 && input.shape()[1] == channels;
    }

    public int filters() {
        return filters;
    }

    public ConvLayer filters(int filters) {
        this.filters = filters;
        return this;
    }

    public int kernelWidth() {
        return kernelWidth;
    }

    public int kernelHeight() {
        return kernelHeight;
    }

    public ConvLayer kernelSize(int kernelWidth, int kernelHeight) {
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        return this;
    }

    public int channels() {
        return channels;
    }

    public ConvLayer channels(int channels) {
        this.channels = channels;
        return this;
    }

    public int stride() {
        return stride;
    }

    public ConvLayer stride(int stride) {
        this.stride = stride;
        return this;
    }

    public int padding() {
        return padding;
    }

    public ConvLayer padding(int padding) {
        this.padding = padding;
        return this;
    }
}
