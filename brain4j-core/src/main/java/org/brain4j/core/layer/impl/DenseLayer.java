package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;

import java.util.Random;

/**
 * Implementation of a fully connected (dense) neural network layer.
 * <p>
 * This layer performs a linear transformation on the input tensor,
 * followed by the application of a specified activation function.
 * </p>
 * <p>Inputs are expected to have the shape <code>[batch, ..., input_size]</code>,
 * outputs have the shape <code>[batch, ..., dimension]</code> where <code>dimension</code>
 * is the amount of neurons in this layer.
 * </p>
 * Weights are represented with the following shapes:
 * <ul>
 *   <li><code>weights</code> has shape <code>[input_size, output_size]</code></li>
 *   <li><code>bias</code> has shape <code>[output_size]</code></li>
 * </ul>
 * @author xEcho1337
 * @since 3.0
 */
public class DenseLayer extends Layer {

    private int dimension;
    
    public DenseLayer() {
    }
    
    /**
     * Constructs a new instance of a dense layer with a linear activation.
     * @param dimension the dimension of the output
     */
    public DenseLayer(int dimension) {
        this(dimension, Activations.LINEAR);
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activations activation) {
        this.dimension = dimension;
        this.activation = activation.function();
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activation activation) {
        this.dimension = dimension;
        this.activation = activation;
    }

    @Override
    public Layer connect(Layer previous) {
        // Shape: [input_size, output_size]
        this.weights = Tensors.zeros(previous.size(), dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        if (input == 0) return;
        
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];
        Tensor[] beforeActivation = new Tensor[inputs.length];
        
        for (int i = 0; i < result.length; i++) {
            Tensor input = inputs[i];

            Tensor output = input
                .matmulGrad(weights)
                .addGrad(bias);

            beforeActivation[i] = output;
            result[i] = activation instanceof LinearActivation
                ? output : output.activateGrad(activation);
        }

        cache.rememberOutput(this, beforeActivation);
        return result;
    }
    
    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
    }
    
    @Override
    public boolean validInput(Tensor input) {
        int[] shape = input.shape();
        int[] weightsShape = weights.shape();

        return shape[shape.length - 1] == weightsShape[0];
    }
}
