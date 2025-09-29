package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;
import java.util.Random;

public class LiquidLayer extends Layer {

    private DenseLayer hiddenParams;
    private DenseLayer tauParams;
    private final int dimension;
    private final int mSteps;
    private double tauMin = 0.5;
    private double tauMax = 5.0;

    public LiquidLayer(int dimension, int mSteps) {
        this.dimension = dimension;
        this.mSteps = mSteps;
    }

    public LiquidLayer(int dimension, int mSteps, double tauMin, double tauMax) {
        this.dimension = dimension;
        this.mSteps = mSteps;
        this.tauMin = tauMin;
        this.tauMax = tauMax;
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
        this.hiddenParams.initWeights(generator, input, output);
        this.tauParams.initWeights(generator, input, output);
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

        for (int t = 0; t < timesteps; t++) {
            Range[] ranges = { Range.all(), Range.point(t), Range.all() };

            Tensor x_t = input.sliceGrad(ranges).squeezeGrad(1); // [B, input_dim]
            Tensor deltaT = deltas.sliceGrad(ranges).squeezeGrad(1); // [B, 1]

            Tensor tau = tauParams.forward(cache, x_t).map(v -> Math.clamp(v, tauMin, tauMax));
            Tensor delta_t = deltaT.divide(mSteps); // [B, 1]

            for (int i = 0; i < mSteps; i++) {
                Tensor projInput = x_t.matmulGrad(weights).addGrad(bias);
                Tensor projHidden = hiddenParams.forward(cache, hidden);

                // z = tanh(Wx + Uh + b)
                Tensor z = projInput.addGrad(projHidden).activateGrad(Activations.TANH.function());
                // (Δt / τ(x))
                Tensor alpha = delta_t.broadcastLike(tau).divGrad(tau); // [B, hidden_dim]

                Tensor deltaH = z.addGrad(hidden.times(-1));
                hidden = hidden.addGrad(alpha.times(deltaH));
            }
        }


        return new Tensor[] { hidden, deltas };
    }

    @Override
    public int size() {
        return dimension;
    }
}
