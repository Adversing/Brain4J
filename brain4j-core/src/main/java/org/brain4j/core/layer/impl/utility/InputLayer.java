package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;

public class InputLayer extends Layer {

    private int[] shape;
    
    public InputLayer() {
    }

    public InputLayer(int... shape) {
        this.shape = shape;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return inputs;
    }

    @Override
    public int size() {
        return shape[shape.length - 1];
    }
    
    @Override
    public void serialize(JsonObject object) {
        JsonArray array = new JsonArray();
        
        for (int dimension : shape) {
            array.add(dimension);
        }
        
        object.addProperty("shape.length", shape.length);
        object.add("shape.data", array);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        JsonArray data = object.getAsJsonArray("shape.data");
        this.shape = new int[data.size()];
        
        for (int i = 0; i < shape.length; i++) {
            shape[i] = data.get(i).getAsInt();
        }
    }
    
    @Override
    public boolean validInput(Tensor input) {
        if (input.rank() > shape.length + 1) return false;

        int[] inputShape = input.shape();
        // Leniency, input COULD have a batch, it's not guaranteed
        int offset = inputShape.length - shape.length;

        for (int i = 0; i < shape.length; i++) {
            if (inputShape[i + offset] != shape[i]) return false;
        }

        return true;
    }
    
    public int[] shape() {
        return shape;
    }
}
