package org.brain4j.core.layer.impl;

import com.google.gson.JsonObject;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public class NormLSTMLayer extends Layer {
    
    private Tensor hiddenWeights;
    private Tensor normGateWeights;
    private Tensor normGateBias;
    private Tensor normCellWeights;
    private Tensor normCellBias;
    
    private int hiddenDimension;
    private boolean returnSequences;

    private NormLSTMLayer() {
    }
    
    public NormLSTMLayer(int hiddenDimension, boolean returnSequences) {
        this.hiddenDimension = hiddenDimension;
        this.returnSequences = returnSequences;
    }
    
    @Override
    public Layer connect(Layer previous) {
        List<Tensor> gates = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            gates.add(Tensors.orthogonal(hiddenDimension, hiddenDimension));
        }

        this.weights = Tensors.zeros(previous.size(), 4 * hiddenDimension).withGrad();
        this.hiddenWeights = Tensors.concat(gates, 1).withGrad();
        this.bias = Tensors.zeros(4 * hiddenDimension).withGrad();

        this.normGateWeights = Tensors.ones(4 * hiddenDimension).withGrad();
        this.normGateBias = Tensors.zeros(4 * hiddenDimension).withGrad();
        this.normCellWeights = Tensors.ones(hiddenDimension).withGrad();
        this.normCellBias = Tensors.zeros(hiddenDimension).withGrad();
        return this;
    }
    
    @Override
    public void initWeights(RandomGenerator generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, 4 * hiddenDimension));

        for (int i = 0; i < hiddenDimension; i++) {
            bias.set(1, i);
        }
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
        
        // [batch, timesteps, 4 * hidden_dim]
        Tensor projection = input.matmulGrad(weights);
        // [batch, timesteps, hidden_size]
        Tensor hiddenState = Tensors.zeros(batch, hiddenDimension).withGrad();
        Tensor cellState = Tensors.zeros(batch, hiddenDimension).withGrad();

        List<Tensor> hiddenStates = new ArrayList<>();
        
        Activation tanh = Activations.TANH.function();
        Activation sigmoid = Activations.SIGMOID.function();
        
        for (int t = 0; t < timesteps; t++) {
            Tensor timestep = projection.sliceGrad(Range.all(), Range.point(t), Range.all()).squeezeGrad(1);
            Tensor hiddenProj = hiddenState.matmulGrad(hiddenWeights); // [batch, 4 * hidden_dim]
            
            Tensor gates = timestep.addGrad(hiddenProj).addGrad(bias); // [batch, 4 * hidden_dim]
            
            gates = gates.layerNorm(1e-5).mulGrad(normGateWeights).addGrad(normGateBias);
            
            Tensor forgetChunk = gates.sliceGrad(Range.all(), Range.interval(0, hiddenDimension));
            Tensor inputChunk = gates.sliceGrad(Range.all(), Range.interval(hiddenDimension, 2 * hiddenDimension));
            Tensor candidateChunk = gates.sliceGrad(Range.all(), Range.interval(2 * hiddenDimension, 3 * hiddenDimension));
            Tensor outputChunk = gates.sliceGrad(Range.all(), Range.interval(3 * hiddenDimension, 4 * hiddenDimension));
            
            Tensor f = forgetChunk.activateGrad(sigmoid);
            Tensor i = inputChunk.activateGrad(sigmoid);
            Tensor g = candidateChunk.activateGrad(tanh);
            Tensor out = outputChunk.activateGrad(sigmoid);
            
            cellState = f.mulGrad(cellState).addGrad(i.mulGrad(g));
            cellState = cellState.layerNorm(1e-5).mulGrad(normCellWeights).addGrad(normCellBias);
            hiddenState = out.mulGrad(cellState.activateGrad(tanh));
            
            if (returnSequences) {
                hiddenStates.add(hiddenState.reshapeGrad(batch, 1, hiddenDimension));
            }
        }
        
        // [batch, timesteps, hidden_dim]
        Tensor result = hiddenState;

        if (returnSequences) {
            result = Tensors.concatGrad(hiddenStates, 1);
        }
        
        return new Tensor[] { result };
    }
    
    @Override
    public int size() {
        return hiddenDimension;
    }
    
    @Override
    public void serialize(JsonObject object) {
        object.addProperty("hidden_dimension", hiddenDimension);
    }
    
    @Override
    public void deserialize(JsonObject object) {
        this.hiddenDimension = object.get("hidden_dimension").getAsInt();
    }
    
    @Override
    public void loadWeights(Map<String, Tensor> mappedWeights) {
        super.loadWeights(mappedWeights);
        this.hiddenWeights = mappedWeights.get("hidden_weights");
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);
        
        Tensor weightsGrad = optimizer.step(hiddenWeights, hiddenWeights.grad());
        Tensor optimizedGateGrad = optimizer.step(normGateWeights, normGateWeights.grad());
        Tensor optimizedCellGrad = optimizer.step(normCellWeights, normCellWeights.grad());
        Tensor gateBiasGrad = normGateBias.grad().sum(0, false);
        Tensor cellBiasGrad = normCellBias.grad().sum(0, false);

        clipper.clip(weightsGrad);
        clipper.clip(optimizedGateGrad);
        clipper.clip(optimizedCellGrad);
        clipper.clip(gateBiasGrad);
        clipper.clip(cellBiasGrad);

        updater.change(hiddenWeights, weightsGrad);
        updater.change(normGateWeights, optimizedGateGrad);
        updater.change(normCellWeights, optimizedCellGrad);
        updater.change(normGateBias, gateBiasGrad);
        updater.change(normCellBias, cellBiasGrad);
    }

    @Override
    public void resetGrad() {
        super.resetGrad();
        hiddenWeights.zerograd();
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
    public Map<String, Tensor> weightsMap() {
        var result = super.weightsMap();
        result.put("hidden_weights", hiddenWeights);
        return result;
    }
}