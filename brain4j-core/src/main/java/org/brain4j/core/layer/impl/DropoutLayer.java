package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Implementation of a dropout layer, used to mitigate overfitting.
 * During training, it randomly turns to zero a fraction of the values in the input tensor.
 * During inference, the input gets multiplied by {@code 1 - dropout}.
 * @author xEcho1337
 */
public class DropoutLayer extends Layer {

    private final RandomGenerator random;
    private double dropoutRate;
    private int size;
    
    public DropoutLayer() {
        this.random = new SplittableRandom();
    }
    
    /**
     * Constructs a new dropout layer instance.
     * @param dropoutRate the dropout rate (0 < dropout < 1), specifying the probability of deactivating each neuron
     * @throws IllegalArgumentException if dropout is outside the range 0-1
     */
    public DropoutLayer(double dropoutRate) {
        if (dropoutRate < 0 || dropoutRate >= 1) {
            throw new IllegalArgumentException("Dropout must be greater than 0 and less than 1!");
        }

        this.random = new SplittableRandom();
        this.dropoutRate = dropoutRate;
    }

    @Override
    public Layer connect(Layer previous) {
        this.size = previous.size();
        return this;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        if (cache.training()) {
            return scale(inputs);
        }
        
        for (Tensor input : inputs) {
            float[] mask = new float[input.elements()];

            for (int i = 0; i < mask.length; i++) {
                if (random.nextDouble() > dropoutRate) continue;

                mask[i] = Float.MIN_VALUE;
            }

            if (input instanceof GpuTensor gpu) {
                gpu.mask(mask);
            }
            
            if (input instanceof CpuTensor cpu) {
                float[] data = cpu.data();
                
                for (int i = 0; i < input.elements(); i++) {
                    data[i] = Math.max(data[i] + mask[i], 0);
                }
            }
        }
        
        return inputs;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void deserialize(JsonObject object) {
        this.dropoutRate = object.get("dropout").getAsDouble();
    }

    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dropout", dropoutRate);
    }
    
    /**
     * Scales the input tensor by {@code 1 - input}.
     * @param inputs the input tensors
     */
    public Tensor[] scale(Tensor... inputs) {
        for (Tensor input : inputs) {
            input.mul(1 - dropoutRate);
        }
        
        return inputs;
    }

    /**
     * Gets the dropout rate
     * @return the dropout rate
     */
    public double dropoutRate() {
        return dropoutRate;
    }
}