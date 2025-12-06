package org.brain4j.math.solver;

import org.brain4j.math.solver.impl.BogackiShampineSolver;
import org.brain4j.math.solver.impl.EulerSolver;
import org.brain4j.math.solver.impl.RungeKuttaSolver;
import org.brain4j.math.tensor.Tensor;

import java.util.function.Function;

/**
 * Contract defining numerical solvers for ordinary differential equations (ODEs),
 * used to advance the hidden state of liquid neural networks through time.
 * <p>
 * Implementations provide different trade-offs between speed and accuracy,
 * such as {@link EulerSolver}, {@link RungeKuttaSolver}, and {@link BogackiShampineSolver}.
 * @author xEcho1337
 */
public interface NumericalSolver {
    /**
     * Performs a single solver step with the selected integration scheme.
     * @param deltaT the timestep delta, shape [batch, 1]
     * @param tauT the projected time constant τ, shape [batch, hidden_dim]
     * @param projInput the projected input (Wx + b) for this timestep, shape [batch, hidden_dim]
     * @param hidden the current hidden state
     * @param hiddenFunction function to project the hidden state (e.g. Uh)
     * @return the next hidden state
     */
    Tensor update(Tensor deltaT, Tensor tauT, Tensor projInput, Tensor hidden, Function<Tensor, Tensor> hiddenFunction);

    /**
     * Resets any internal state or cache maintained by the solver.
     * <p>
     * Some solvers (e.g., {@link BogackiShampineSolver}) cache intermediate values
     * between steps for optimization purposes (such as the FSAL property). This method
     * should be called when starting a new sequence or when the batch size changes
     * to ensure the solver starts with a clean state.
     * <p>
     * The default implementation does nothing, as most solvers are stateless.
     */
    default void resetCache() {
        // no-op by default; stateless solvers don't need to reset anything
    }
}
