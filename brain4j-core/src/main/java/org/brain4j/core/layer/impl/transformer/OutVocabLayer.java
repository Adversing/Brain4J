package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.importing.proto.SerializeUtils;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class OutVocabLayer extends Layer {

    private int vocabSize;
    private int dimension;
    
    public OutVocabLayer() {
    }
    
    public OutVocabLayer(int vocabSize, int dimension, double temperature) {
        this.vocabSize = vocabSize;
        this.dimension = dimension;
        this.activation = new SoftmaxActivation(Math.max(1e-15, temperature));
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(dimension, vocabSize).withGrad();
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.vocabSize = SerializeUtils.attribute(layer, "vocab_size", 0);
        this.dimension = SerializeUtils.attribute(layer, "dimension", 0);
        
        for (ProtoModel.Tensor tensor : tensors) {
            if (tensor.getName().contains("weight")) {
                this.weights = SerializeUtils.deserializeTensor(tensor);
            }
        }
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("vocab_size", SerializeUtils.value(vocabSize));
        builder.putAttrs("dimension", SerializeUtils.value(dimension));
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of(
            SerializeUtils.serializeTensor("weight", weights)
        );
    }
    
    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                    "Expected input with shape [batch_size, seq_length, dimension], got: " + Arrays.toString(shape)
            );
        }
        
        Tensor output = input.matmulGrad(weights);
        Tensor activated = output.activateGrad(activation);

        cache.rememberOutput(this, output);

        return activated;
    }

    @Override
    public int size() {
        return vocabSize;
    }
}
