package org.brain4j.core.layer.impl;

import org.brain4j.common.Tensors;
import org.brain4j.common.activation.Activation;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.index.Range;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;

import java.util.List;
import java.util.Random;

public class LSTMLayer extends Layer {
    
    private Tensor hiddenWeights;
    private int dimension;
    private int hiddenDimension;
    
    private LSTMLayer() {
    }
    
    public LSTMLayer(int dimension, int hiddenDimension) {
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
    }
    
    @Override
    public Layer connect(Layer previous) {
        int size = previous == null ? dimension : previous.size();
        this.weights = Tensors.zeros(size, 4 * hiddenDimension).withGrad();
        this.hiddenWeights = Tensors.zeros(hiddenDimension, 4 * hiddenDimension).withGrad();
        this.bias = Tensors.zeros(4 * hiddenDimension).withGrad();
        return this;
    }
    
    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, 4 * hiddenDimension));
        this.hiddenWeights.map(x -> weightInit.generate(generator, hiddenDimension, 4 * hiddenDimension));
    }
    
    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        // [batch_size, timesteps, dimension]
        if (input.rank() > 3) {
            throw new IllegalArgumentException("Recurrent layers expected 3-dimensional tensors! Got " + input.rank() + "instead");
        }
        
        while (input.rank() < 3) {
            input = input.unsqueeze();
        }
        
        int batch = input.shape()[0];
        int timesteps = input.shape()[1];
        
        // [batch, timesteps, dimension] x [dimension, 4 * hidden_dim]
        // = [batch, timesteps, 4 * hidden_dim]
        Tensor projection = input.matmulGrad(weights);
        
        // [batch_size, timesteps, hidden_size]
        Tensor hiddenState = Tensors.zeros(batch, hiddenDimension).withGrad();
        Tensor cellState = Tensors.zeros(batch, hiddenDimension).withGrad();
        
        Tensor[] hiddenStates = new Tensor[timesteps];
        
        Activation tanh = Activations.TANH.function();
        Activation sigmoid = Activations.SIGMOID.function();
        
        for (int t = 0; t < timesteps; t++) {
            Tensor timestep = projection.sliceGrad(Range.all(), Range.point(t), Range.all()).squeeze(1);
            Tensor hiddenProj = hiddenState.matmulGrad(hiddenWeights); // [batch, 4 * hidden_dim]
            
            Tensor preActivation = timestep.addGrad(hiddenProj).addGrad(bias); // [batch, 4 * hidden_dim]
            
            Tensor forgetChunk = preActivation.sliceGrad(Range.all(), Range.interval(0, hiddenDimension));
            Tensor inputChunk = preActivation.sliceGrad(Range.all(), Range.interval(hiddenDimension, 2 * hiddenDimension));
            Tensor candidateChunk = preActivation.sliceGrad(Range.all(), Range.interval(2 * hiddenDimension, 3 * hiddenDimension));
            Tensor outputChunk = preActivation.sliceGrad(Range.all(), Range.interval(3 * hiddenDimension, 4 * hiddenDimension));
            
            Tensor f = forgetChunk.activateGrad(sigmoid);
            Tensor i = inputChunk.activateGrad(sigmoid);
            Tensor g = candidateChunk.activateGrad(tanh);
            Tensor out = outputChunk.activateGrad(sigmoid);
            
            cellState = f.mulGrad(cellState).addGrad(i.mulGrad(g));
            hiddenState = out.mulGrad(cellState.activateGrad(tanh));
            
            hiddenStates[t] = hiddenState.reshapeGrad(batch, 1, hiddenDimension);
        }
        
        // [batch_size, timesteps, hidden_dim]
        return Tensors.concatGradLast(List.of(hiddenStates), 1);
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);

        Tensor weightsGrad = optimizer.step(hiddenWeights, hiddenWeights.grad());
        
        clipper.clip(weightsGrad);
        updater.change(hiddenWeights, weightsGrad);
    }
    
    @Override
    public int totalBiases() {
        return bias.elements();
    }
    
    @Override
    public int totalWeights() {
        return weights.elements() + hiddenWeights.elements();
    }
    
    @Override
    public int size() {
        return dimension;
    }
}