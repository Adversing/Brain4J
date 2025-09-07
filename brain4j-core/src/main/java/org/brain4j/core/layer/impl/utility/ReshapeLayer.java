package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ReshapeLayer extends Layer {

    private int[] shape;
    
    public ReshapeLayer() {
    }
    
    public ReshapeLayer(int... shape) {
        this.shape = shape;
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < result.length; i++) {
            Tensor input = inputs[i];

            int[] inputShape = input.shape();
            int[] newShape = new int[shape.length + 1];

            newShape[0] = inputShape[0];
            System.arraycopy(shape, 0, newShape, 1, shape.length);

            result[i] = input.reshapeGrad(newShape);
        }

        return result;
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
}
