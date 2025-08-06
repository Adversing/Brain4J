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
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("filters", value(filters));
        layerBuilder.putAttrs("kernelWidth", value(kernelWidth));
        layerBuilder.putAttrs("kernelHeight", value(kernelHeight));
        layerBuilder.putAttrs("stride", value(stride));
        layerBuilder.putAttrs("padding", value(padding));
        return List.of(serializeTensor("weight", weights), serializeTensor("bias", bias));
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
