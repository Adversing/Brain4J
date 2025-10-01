package org.brain4j.core.layer.impl.liquid;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.liquid.solver.NumericalSolver;
import org.brain4j.core.layer.impl.liquid.solver.impl.EulerSolver;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LiquidLayer extends Layer {

    private final int dimension;
    private final double tauMin;
    private final double tauMax;
    private final boolean returnSequences;

    private DenseLayer hiddenParams;
    private DenseLayer tauParams;
    private NumericalSolver solver;

    public LiquidLayer(int dimension, int mSteps, boolean returnSequences) {
        this(dimension, 0.5, 5.0, returnSequences, new EulerSolver(mSteps));
    }

    public LiquidLayer(int dimension, boolean returnSequences, NumericalSolver solver) {
        this(dimension, 0.5, 5.0, returnSequences, solver);
    }

    public LiquidLayer(int dimension, double tauMin, double tauMax, boolean returnSequences, NumericalSolver solver) {
        this.solver = solver;
        this.dimension = dimension;
        this.tauMin = tauMin;
        this.tauMax = tauMax;
        this.returnSequences = returnSequences;
    }

    @Override
    public LiquidLayer connect(Layer previous) {
        this.weights = Tensors.zeros(previous.size(), dimension);
        this.bias = Tensors.zeros(dimension);
        this.hiddenParams = new DenseLayer(dimension).connect(this);
        this.tauParams = new DenseLayer(dimension, Activations.SOFTPLUS).connect(previous);
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(_ -> weightInit.generate(generator, input, output));
        this.hiddenParams.initWeights(generator, dimension, dimension);
        this.tauParams.initWeights(generator, input, dimension);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        checkInputLength(2, inputs);

        Tensor input = inputs[0];
        Tensor deltas = inputs[1];

        if (input.rank() != 3) {
            throw new IllegalArgumentException(
                "LTC requires rank 3 input! Expected [batch, timesteps, features], got: " + Arrays.toString(input.shape())
            );
        }

        if (deltas.rank() != 3) {
            throw new IllegalArgumentException(
                "LTC requires rank 3 deltas! Expected [batch, timesteps, features], got: " + Arrays.toString(deltas.shape())
            );
        }

        int batch = input.shape(0);
        int timesteps = input.shape(1);

        Tensor hidden = Tensors.zeros(batch, dimension).withGrad(); // [B, hidden_dim]

        if (input instanceof GpuTensor gpu) {
            hidden = hidden.to(gpu.device()).withGrad();
        }

        List<Tensor> hiddenStates = new ArrayList<>();

        Tensor projTau = tauParams.forward(cache, input).map(v -> Math.clamp(v, tauMin, tauMax)); // [batch, timesteps, hidden_dim]
        Tensor projInput = input.matmulGrad(weights).addGrad(bias); // [batch, timesteps, hidden_dim]

        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = { Range.all(), Range.point(t), Range.all() };

            Tensor deltaT = deltas.sliceGrad(ranges).squeezeGrad(1); // [batch, 1]
            Tensor tau_t = projTau.sliceGrad(ranges).squeezeGrad(1); // [batch, hidden_dim]
            Tensor projInput_t = projInput.sliceGrad(ranges).squeezeGrad(1); // [batch, hidden_dim]

            hidden = solver.update(deltaT, tau_t, projInput_t, hidden, x -> hiddenParams.forward(cache, x));
            hiddenStates.add(hidden);
        }

        if (returnSequences) {
            hidden = Tensors.concatGrad(hiddenStates);
        }

        return new Tensor[] { hidden, deltas };
    }
    
    @Override
    public void backward(StatesCache cache, Updater updater, Optimizer optimizer) {
        super.backward(cache, updater, optimizer);
        this.hiddenParams.backward(cache, updater, optimizer);
        this.tauParams.backward(cache, updater, optimizer);
    }

    @Override
    public void resetGrad() {
        super.resetGrad();
        this.hiddenParams.resetGrad();
        this.tauParams.resetGrad();
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public int totalWeights() {
        return weights.elements() + hiddenParams.totalWeights() + tauParams.totalWeights();
    }

    @Override
    public int totalBiases() {
        return bias.elements() + hiddenParams.totalBiases() + tauParams.totalBiases();
    }

    public NumericalSolver solver() {
        return solver;
    }

    public LiquidLayer solver(NumericalSolver solver) {
        this.solver = solver;
        return this;
    }

    public DenseLayer hiddenParams() {
        return hiddenParams;
    }
    
    public void setHiddenParams(DenseLayer hiddenParams) {
        this.hiddenParams = hiddenParams;
    }
    
    public DenseLayer tauParams() {
        return tauParams;
    }
    
    public void setTauParams(DenseLayer tauParams) {
        this.tauParams = tauParams;
    }
    
    public int dimension() {
        return dimension;
    }

    public double tauMin() {
        return tauMin;
    }
    
    public double tauMax() {
        return tauMax;
    }
}
