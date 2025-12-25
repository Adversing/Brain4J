package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * Implementation of a layer normalization layer,
 * used to normalize inputs and improve training.
 * <h2>Shape conventions:</h2>
 * <ul>
 *     <li>Input: {@code [batch, ..., input_dim]}</li>
 *     <li>Output: {@code [batch, ..., input_dim]}</li>
 *     <li>Weights: {@code [input_dim]}</li>
 *     <li>Bias: {@code [input_dim]}</li>
 * </ul>
 * @author xEcho1337
 */
public class NormLayer extends Layer {

    private double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon.
     */
    public NormLayer() {
        this(1e-5);
    }
    
    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public NormLayer(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.ones(previous.size()).withGrad();
        this.bias = Tensors.zeros(previous.size()).withGrad();
        
        return this;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);

        Tensor input = inputs[0];
        Tensor cloned = input.clone();
        cloned.setAutogradContext(input.autogradContext());

        Tensor result = cloned.layerNorm(epsilon).mulGrad(weights).addGrad(bias);
        return new Tensor[] { result };
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("epsilon", epsilon);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.epsilon = object.get("epsilon").getAsDouble();
    }
    
    @Override
    public int size() {
        return weights.elements();
    }

    public double getEpsilon() {
        return epsilon;
    }

    public NormLayer setEpsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }
}
