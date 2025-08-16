package org.brain4j.core.layer.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * Implementation of a dropout layer, it's used to mitigate overfitting by randomly deactivating a part of the neurons
 * during training. When inferencing, the input gets scaled by {@code 1 - dropout}.
 * @author xEcho1337
 */
public class DropoutLayer extends Layer {

    private final Random random;
    private double dropoutRate;
    
    public DropoutLayer() {
        this.random = Random.from(new SplittableRandom());
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

        this.random = Random.from(new SplittableRandom());
        this.dropoutRate = dropoutRate;
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.dropoutRate = SerializeUtils.attribute(layer, "dropout_rate", 0.0);
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("dropout_rate", SerializeUtils.value(dropoutRate));
    }
    
    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        if (training) {
            return scale(input);
        }

        for (int i = 0; i < input.elements(); i++) {
            if (random.nextDouble() > dropoutRate) {
                continue;
            }

            input.data()[i] = 0;
        }

        return input;
    }

    @Override
    public int size() {
        return 0;
    }

    /**
     * Scales the input tensor by {@code 1 - input}.
     * @param input the input tensor
     * @return the scaled tensor
     */
    public Tensor scale(Tensor input) {
        return input.mul(1 - dropoutRate);
    }

    /**
     * Gets the dropout rate
     * @return the dropout rate
     */
    public double dropoutRate() {
        return dropoutRate;
    }
}