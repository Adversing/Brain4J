package org.brain4j.core.layer.impl.utility;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;

/**
 * Input layer used to define the expected shape of data entering the network.
 * <p>
 * This layer must always be placed as the first layer in a model.
 * It does not perform any computation but acts as a contract,
 * ensuring that tensors flowing into the model have the correct shape.
 * Only the last {@code N} dimensions are strictly checked, where
 * {@code N} is the rank of the specified input shape.
 * This allows an optional batch
 * dimension to be present without affecting validation.
 * </p>
 * <h2>Use example:</h2>
 * <blockquote><pre>{@code
 * Model model = Sequential.of(
 *     // Expects input with shape [..., 3, 4]
 *     new InputLayer(3, 4),
 *     new DenseLayer(16),
 *     new DenseLayer(10)
 * );
 * </pre></blockquote>
 */
public class InputLayer extends Layer {

    private int[] shape;

    protected InputLayer() {
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
