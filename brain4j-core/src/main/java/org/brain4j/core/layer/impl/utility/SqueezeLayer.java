package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;

public class SqueezeLayer extends Layer {
    
    private int dimension;
    private int size;
    
    public SqueezeLayer(int dimension) {
        this.dimension = dimension;
    }
    
    @Override
    public Layer connect(Layer previous) {
        this.size = previous.size();
        return this;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] results = new Tensor[inputs.length];

        for (int i = 0; i < results.length; i++) {
            Tensor input = inputs[i];
            results[i] = dimension == -1 ? input.squeeze() : input.squeeze(dimension);
        }

        return results;
    }
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
        object.addProperty("size", size);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
        this.size = object.get("size").getAsInt();
    }
}
