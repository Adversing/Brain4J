package org.brain4j.core.layer.impl.liquid.solver.impl;

import org.brain4j.core.layer.impl.liquid.solver.NumericalSolver;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

import java.util.function.Function;

/**
 * Implements the classical fourth-order Runge–Kutta (RK4) method for integrating
 * ordinary differential equations (ODEs) of the form:
 *
 * <pre>
 *     dh/dt = F(h, t)
 * </pre>
 *
 * Unlike the simple Euler method, which estimates the next state using only the
 * slope at the beginning of the interval, RK4 evaluates the slope at four points
 * within the timestep and combines them into a weighted average. This provides
 * a significantly more accurate and stable integration without requiring
 * additional substeps.
 *
 * <p>In the context of liquid neural networks, {@code RungeKuttaSolver} advances
 * the hidden state of each neuron over a timestep {@code Δt}, taking into account
 * both its internal dynamics and external inputs. The solver can typically operate
 * with {@code mSteps = 1}, as the RK4 formulation already captures intermediate
 * behavior within the interval.</p>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * NumericalSolver solver = new RungeKuttaSolver();
 * Tensor nextHidden = solver.update(deltaT, tau, projInput, hidden, x -> hiddenParams.forward(cache, x));
 * }</pre>
 *
 * @see EulerSolver
 * @author xEcho1337
 */
public class RungeKuttaSolver implements NumericalSolver {

    @Override
    public Tensor update(Tensor deltaTimestep, Tensor tauT, Tensor projInput,
                         Tensor hidden, Function<Tensor, Tensor> hiddenFunction) {

        // Δt / τ(x)
        Tensor alpha = deltaTimestep.broadcastLike(tauT).divGrad(tauT);
        // k1 = F(h, t)
        Tensor k1 = computeF(hidden, projInput, hiddenFunction);
        // k2 = F(h + 0.5*Δt*k1, t + Δt/2)
        Tensor k2 = computeF(hidden.addGrad(alpha.times(k1).times(0.5f)),
            projInput, hiddenFunction);
        // k3 = F(h + 0.5*Δt*k2, t + Δt/2)
        Tensor k3 = computeF(hidden.addGrad(alpha.times(k2).times(0.5f)),
            projInput, hiddenFunction);
        // k4 = F(h + Δt*k3, t + Δt)
        Tensor k4 = computeF(hidden.addGrad(alpha.times(k3)),
            projInput, hiddenFunction);
        // h_next = h + (Δt/6)*(k1 + 2*k2 + 2*k3 + k4)
        Tensor increment = k1.addGrad(k2.times(2)).addGrad(k3.times(2)).addGrad(k4)
            .times(1.0f / 6.0f);
        hidden = hidden.addGrad(alpha.times(increment));

        return hidden;
    }

    private Tensor computeF(Tensor h, Tensor projInput, Function<Tensor, Tensor> hiddenFunction) {
        Tensor projHidden = hiddenFunction.apply(h);
        Tensor z = projInput.addGrad(projHidden).activateGrad(Activations.TANH.function());
        return z.addGrad(h.times(-1)); // = z - h
    }
}