package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;

import static org.brain4j.core.importing.Registries.SCALER_REGISTRY;

public class ScalingLayer extends Layer {

    private FeatureScaler scaler;
    private int dimension;

    private ScalingLayer() {
    }

    public ScalingLayer(FeatureScaler scaler) {
        this.scaler = scaler;
    }

    @Override
    public void connect(Layer previous) {
        this.dimension = previous.size();

    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] outputs = new Tensor[inputs.length];

        for (int i = 0; i < outputs.length; i++) {
            Tensor input = inputs[i];
            Tensor result = scaler.transform(input);

            AutogradContext context = input.getAutogradContext();
            result.setAutogradContext(context);

            outputs[i] = result;
        }

        return outputs;
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("scaler", SCALER_REGISTRY.fromClass(scaler.getClass()));
        scaler.serialize(object);
    }

    @Override
    public void deserialize(JsonObject object) {
        String scalerType = object.getAsJsonPrimitive("scaler").getAsString();
        scaler = SCALER_REGISTRY.toInstance(scalerType);
        scaler.deserialize(object);
    }
}
