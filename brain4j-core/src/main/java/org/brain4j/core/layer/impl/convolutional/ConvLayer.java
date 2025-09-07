package org.brain4j.core.layer.impl.convolutional;

import com.google.gson.JsonObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ConvLayer extends Layer {

    private int channels;
    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int stride = 1;
    private int padding = 0;
    
    public ConvLayer() {
    }
    
    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight) {
        this.activation = activation.function();
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
    }
    
    @Override
    public Layer connect(Layer previous) {
        this.channels = previous.size();
        this.bias = Tensors.zeros(filters).withGrad();
        this.weights = Tensors.zeros(filters, channels, kernelHeight, kernelWidth).withGrad();
        
        return this;
    }
    
    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (!validInput(input)) {
            throw new IllegalArgumentException("Input dimension mismatch! Got: " + Arrays.toString(input.shape()));
        }

        Tensor convolved = input.convolveGrad(weights);
        Tensor added = convolved.addGrad(bias.reshape(1, filters, 1, 1));

        return new Tensor[] { added.activateGrad(activation) };
    }

    @Override
    public int size() {
        return filters;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("channels", channels);
        object.addProperty("filters", filters);
        object.addProperty("kernel_width", kernelWidth);
        object.addProperty("kernel_height", kernelHeight);
        object.addProperty("stride", stride);
        object.addProperty("padding", padding);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.channels = object.get("channels").getAsInt();
        this.filters = object.get("filters").getAsInt();
        this.kernelWidth = object.get("kernel_width").getAsInt();
        this.kernelHeight = object.get("kernel_height").getAsInt();
        this.stride = object.get("stride").getAsInt();
        this.padding = object.get("padding").getAsInt();
    }
    
    @Override
    public boolean validInput(Tensor input) {
        // [batch_size, channels, height, width]
        return input.rank() == 4 && input.shape(1) == channels;
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
