package org.brain4j.core.layer.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.index.Range;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Implementation of a recurrent layer.
 * @apiNote This implementation is not completed and doesn't support training yet.
 * @author xEcho1337
 * @since 3.0
 */
public class RecurrentLayer extends Layer {

    private final int dimension;
    private final int hiddenDimension;
    private Tensor inputWeights;
    private Tensor hiddenWeights;
    private Tensor hiddenBias;

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
        this.hiddenWeights.map(x -> weightInit.generate(generator, input, output));
        this.hiddenBias.map(x -> weightInit.generate(generator, input, output));
        this.weights.map(x -> weightInit.generate(generator, input, output));
        this.bias.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        // [batch_size, timesteps, dimension]
        Tensor input = context.input();
        int batch = input.shape()[0];
        int timesteps = input.shape()[1];

        // [batch_size, timesteps, hidden_size]
        Tensor projectedInput = input.matmulGrad(inputWeights);
        Tensor hiddenState = Tensors.zeros(batch, hiddenDimension).withGrad();

        Tensor[] allStates = new Tensor[timesteps];

        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = new Range[] { Range.all(), Range.point(t), Range.all() };

            Tensor timestepX = projectedInput.slice(ranges).squeeze(1);
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
    public boolean validateInput(Tensor input) {
        return input.rank() == 3;
    }

    @Override
    public int size() {
        return dimension;
    }
}
