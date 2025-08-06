package org.brain4j.core.layer.impl.transformer;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.activation.impl.SoftmaxActivation;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class OutVocabLayer extends Layer {

    private int vocabSize;
    private int dimension;

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
        this.vocabSize = attribute(layer, "vocab_size", 0);
        this.dimension = attribute(layer, "dimension", 0);
        
        for (ProtoModel.Tensor tensor : tensors) {
            if (tensor.getName().equals("weight")) {
                this.weights = deserializeTensor(tensor);
            }
        }
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> serialize(ProtoModel.Layer.Builder layerBuilder) {
        layerBuilder.putAttrs("vocab_size", value(vocabSize));
        layerBuilder.putAttrs("dimension", value(dimension));
        return List.of(serializeTensor("weight", weights));
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                    "Expected input with shape [batch_size, seq_length, dimension], got: " + Arrays.toString(shape)
            );
        }

        StatesCache cache = context.cache();

        Tensor output = input.matmulGrad(weights);
        Tensor activated = output.activateGrad(activation);

        cache.setPreActivation(this, output);

        return activated;
    }

    @Override
    public int size() {
        return vocabSize;
    }
}
