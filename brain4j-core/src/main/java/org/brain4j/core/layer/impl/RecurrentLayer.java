package org.brain4j.core.layer.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.index.Range;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.importing.proto.ProtoModel;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.List;
import java.util.Random;

/**
 * Implementation of a recurrent layer.
 * @apiNote This implementation is not completed and doesn't support training yet.
 * @author xEcho1337
 * @since 3.0
 */
public class RecurrentLayer extends Layer {

    private int dimension;
    private int hiddenDimension;
    private Tensor inputWeights;
    private Tensor hiddenWeights;
    private Tensor hiddenBias;
    
    public RecurrentLayer() {
    }
    
    /**
     * Constructs a new recurrent layer instance.
     *
     * @param dimension the dimension of the output
     * @param hiddenDimension the dimension of the hidden states
     * @param activation the activation function
     */
    public RecurrentLayer(int dimension, int hiddenDimension, Activations activation) {
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
        this.activation = activation.function();
        this.weightInit = this.activation.defaultWeightInit();
    }

    @Override
    public Layer connect(Layer previous) {
        int size = previous == null ? dimension : previous.size();

        this.inputWeights = Tensors.zeros(size, hiddenDimension).withGrad();
        this.hiddenWeights = Tensors.zeros(hiddenDimension, hiddenDimension).withGrad();
        this.hiddenBias = Tensors.zeros(hiddenDimension).withGrad();

        this.weights = Tensors.zeros(hiddenDimension, dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.inputWeights.map(x -> weightInit.generate(generator, input, output));
        this.hiddenWeights.map(x -> weightInit.generate(generator, hiddenDimension, hiddenDimension));
        this.hiddenBias.map(x -> weightInit.generate(generator, hiddenDimension, hiddenDimension));
        this.weights.map(x -> weightInit.generate(generator, hiddenDimension, output));
        this.bias.map(x -> weightInit.generate(generator, input, output));
    }
    
    @Override
    public void deserialize(List<ProtoModel.Tensor> tensors, ProtoModel.Layer layer) {
        this.dimension = attribute(layer, "dimension", 0);
        this.hiddenDimension = attribute(layer, "hidden_dimension", 0);
        
        for (ProtoModel.Tensor tensor : tensors) {
            String name = tensor.getName().split("\\.")[2];
            switch (name) {
                case "input_weight" -> this.inputWeights = deserializeTensor(tensor);
                case "hidden_weight" -> this.hiddenWeights = deserializeTensor(tensor);
                case "hidden_bias" -> this.hiddenBias = deserializeTensor(tensor);
                case "weights" -> this.weights = deserializeTensor(tensor);
                case "bias" -> this.bias = deserializeTensor(tensor);
            }
        }
    }
    
    @Override
    public void serialize(ProtoModel.Layer.Builder builder) {
        builder.putAttrs("dimension", value(dimension));
        builder.putAttrs("hidden_dimension", value(hiddenDimension));
        builder.putAttrs("activation", value(activation.name()));
    }
    
    @Override
    public Tensor forward(ForwardContext context) {
        // [batch_size, timesteps, dimension]
        Tensor input = context.input();
        
        if (input.rank() > 3) {
            throw new IllegalArgumentException("Recurrent layers expected 3-dimensional tensors! Got " + input.rank() + "instead");
        }
        
        while (input.rank() < 3) {
            input = input.unsqueeze();
        }
        
        int batch = input.shape()[0];
        int timesteps = input.shape()[1];

        // [batch_size, timesteps, hidden_size]
        Tensor projectedInput = input.matmulGrad(inputWeights);
        Tensor hiddenState = Tensors.zeros(batch, hiddenDimension).withGrad();

        Tensor[] allStates = new Tensor[timesteps];

        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = new Range[] { Range.all(), Range.point(t), Range.all() };

            Tensor timestepX = projectedInput.sliceGrad(ranges).squeeze(1);
            Tensor timestepH = hiddenState.matmulGrad(hiddenWeights);

            hiddenState = timestepX.addGrad(timestepH).addGrad(hiddenBias).activateGrad(activation);
            allStates[t] = hiddenState.reshapeGrad(batch, 1, hiddenDimension);
        }
        
        // [batch_size, timesteps, hidden_dim]
        Tensor sequence = Tensors.concatGrad(List.of(allStates), 1);
        Tensor output = sequence.matmulGrad(weights).addGrad(bias);
        
        context.cache().setPreActivation(this, output);
        return output;
    }
    
    @Override
    public void backward(Updater updater, Optimizer optimizer, int index) {
        super.backward(updater, optimizer, index);
        
        Tensor inputWeightsGrad = optimizer.step(inputWeights, inputWeights.grad());
        Tensor hiddenWeightsGrad = optimizer.step(hiddenWeights, hiddenWeights.grad());
        Tensor hiddenBiasGrad = hiddenBias.grad().sum(0, false);
        
        clipper.clip(inputWeightsGrad);
        clipper.clip(hiddenWeightsGrad);
        clipper.clip(hiddenBiasGrad);
        
        updater.change(inputWeights, inputWeightsGrad);
        updater.change(hiddenWeights, hiddenWeightsGrad);
        updater.change(hiddenBias, hiddenBiasGrad);
    }
    
    @Override
    public boolean validateInput(Tensor input) {
        return input.rank() == 3;
    }

    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public List<ProtoModel.Tensor.Builder> weightsList() {
        return List.of(
            serializeTensor("input_weight", inputWeights),
            serializeTensor("hidden_weight", hiddenWeights),
            serializeTensor("hidden_bias", hiddenBias),
            serializeTensor("weights", weights),
            serializeTensor("bias", bias)
        );
    }
}
