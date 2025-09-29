package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;

public class SqueezeLayer extends Layer {
    
    private int dimension;

    public SqueezeLayer(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] results = new Tensor[inputs.length];

        for (int i = 0; i < results.length; i++) {
            Tensor input = inputs[i];
            results[i] = dimension == -1 ? input.squeezeGrad() : input.squeezeGrad(dimension);
        }

        return results;
    }
    
    @Override
    public int size() {
        return 0;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
    }
}
