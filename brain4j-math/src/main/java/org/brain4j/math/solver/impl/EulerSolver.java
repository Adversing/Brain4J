package org.brain4j.math.solver.impl;

import org.brain4j.math.solver.NumericalSolver;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

import java.util.function.Function;

/**
 * Implements the explicit Euler method for integrating ordinary differential
 * equations (ODEs) of the form:
 *
 * <pre>
 *     dh/dt = F(h, t)
 * </pre>
 *
 * The Euler scheme advances the hidden state using a single slope evaluation
 * at the beginning of each interval. Since this method is only first-order
 * accurate, the integration error can accumulate quickly unless the timestep
 * is kept very small.
 *
 * <p>To mitigate this, the solver allows splitting each timestep into multiple
 * substeps ({@code mSteps}). Each substep integrates with a reduced delta
 * ({@code Δt / mSteps}), improving stability at the cost of additional
 * function evaluations.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * NumericalSolver solver = new EulerSolver(8); // 8 micro-steps per timestep
 * Tensor nextHidden = solver.update(deltaT, tau, projInput, hidden, hiddenParams::forward);
 * }</pre>
 *
 * <p>This solver is simple and efficient but less accurate than
 * {@link RungeKuttaSolver}, which typically achieves higher precision
 * without requiring substeps.</p>
 *
 * @see RungeKuttaSolver
 * @author xEcho1337
 */
public record EulerSolver(int mSteps) implements NumericalSolver {

    @Override
    public Tensor update(Tensor deltaTimestep, Tensor tauT, Tensor projInput, Tensor hidden, Function<Tensor, Tensor> hiddenFunction) {
        Tensor deltaNorm = deltaTimestep.divide(mSteps).withGrad();
        // (Δt / τ(x))
        Tensor deltaTime = deltaNorm.broadcastLike(tauT).divGrad(tauT); // [batch, hidden_dim]

        for (int i = 0; i < mSteps; i++) {
            Tensor projHidden = hiddenFunction.apply(hidden);
            // z = tanh(Wx + Uh + b)
            Tensor z = projInput.addGrad(projHidden).activateGrad(Activations.TANH.function());
            Tensor deltaH = z.addGrad(hidden.mul(-1));

            hidden = hidden.addGrad(deltaTime.mulGrad(deltaH));
        }

        return hidden;
    }
}
