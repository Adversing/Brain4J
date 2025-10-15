package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * Root Mean Square Normalization (RMSNorm)
 * Normalizes only by the root-mean-square, without centering (no mean subtraction).
 * <p>
 * Formula:
 *   y = x / sqrt(mean(x^2) + eps) * gamma
 *
 * <h2>Shape conventions:</h2>
 * <ul>
 *     <li>Input:  [batch, ..., hidden_dim]</li>
 *     <li>Output: [batch, ..., hidden_dim]</li>
 *     <li>Weights: [hidden_dim]</li>
 * </ul>
 *
 * This is used in modern LLMs (e.g., LLaMA, Qwen, Mistral)
 * instead of LayerNorm for improved numerical stability.
 *
 * @author xEcho1337
 */
public class RMSNormLayer extends Layer {

    private double epsilon;

    public RMSNormLayer() {
        this(1e-6);
    }

    public RMSNormLayer(double epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.ones(previous.size()).withGrad();
        return this;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(1, inputs);

        Tensor input = inputs[0];
        // x / sqrt(mean(x^2) + eps)
        Tensor rms = input.pow(2).mean(-1, true).add(epsilon).sqrt();
        Tensor norm = input.div(rms).mulGrad(weights);

        return new Tensor[] { norm };
    }

    @Override
    public int size() {
        return weights.elements();
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("epsilon", epsilon);
    }

    @Override
    public void deserialize(JsonObject object) {
        this.epsilon = object.get("epsilon").getAsDouble();
    }

    public double epsilon() {
        return epsilon;
    }

    public RMSNormLayer epsilon(double epsilon) {
        this.epsilon = epsilon;
        return this;
    }
}
