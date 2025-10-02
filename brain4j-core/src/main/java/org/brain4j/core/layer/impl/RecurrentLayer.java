package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Implementation of a recurrent layer.
 * @apiNote This implementation is not completed and doesn't support training yet.
 * @author xEcho1337
 */
public class RecurrentLayer extends Layer {

    private Tensor inputWeights;
    private Tensor hiddenWeights;
    private Tensor hiddenBias;
    private int dimension;
    private int hiddenDimension;
    
    private RecurrentLayer() {
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
        this.hiddenWeights = Tensors.orthogonal(hiddenDimension, hiddenDimension).withGrad();
        this.hiddenBias = Tensors.zeros(hiddenDimension).withGrad();
        this.weights = Tensors.zeros(hiddenDimension, dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.inputWeights.map(_ -> weightInit.generate(generator, input, output));
        this.weights.map(_ -> weightInit.generate(generator, hiddenDimension, output));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        // [batch, timesteps, dimension]
        Tensor input = inputs[0];

        if (input.rank() > 3) {
            throw new IllegalArgumentException("Recurrent layers expected 3-dimensional tensors! Got " + input.rank() + "instead");
        }
        
        while (input.rank() < 3) {
            input = input.unsqueeze();
        }
        
        int batch = input.shape(0);
        int timesteps = input.shape(1);

        // [batch, timesteps, hidden_size]
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
        
        // [batch, timesteps, hidden_dim]
        Tensor sequence = Tensors.concatGrad(List.of(allStates), 1);
        Tensor output = sequence.matmulGrad(weights).addGrad(bias);
        
        cache.rememberOutput(this, output);
        return new Tensor[] { output };
    }

    @Override
    public int size() {
        return dimension;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("dimension", dimension);
        object.addProperty("hidden_dimension", hiddenDimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.dimension = object.get("dimension").getAsInt();
        this.hiddenDimension = object.get("hidden_dimension").getAsInt();
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        super.loadWeights(mappedWeights);
        this.inputWeights = mappedWeights.get("input_weights");
        this.hiddenWeights = mappedWeights.get("hidden_weights");
        this.hiddenBias = mappedWeights.get("hidden_bias");
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);
        
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
    public boolean validInput(Tensor input) {
        return input.rank() == 3;
    }

    @Override
    public void resetGrad() {
        super.resetGrad();
        inputWeights.zerograd();
        hiddenWeights.zerograd();
        hiddenBias.zerograd();
    }
    
    @Override
    public int totalBiases() {
        return hiddenBias.elements() + bias.elements();
    }
    
    @Override
    public int totalWeights() {
        return weights.elements() + inputWeights.elements() + hiddenWeights.elements();
    }
    
    @Override
    public Map<String, Tensor> weightsMap() {
        var result = super.weightsMap();
        result.put("input_weights", inputWeights);
        result.put("hidden_weights", hiddenWeights);
        result.put("hidden_bias", hiddenBias);
        return result;
    }
}
