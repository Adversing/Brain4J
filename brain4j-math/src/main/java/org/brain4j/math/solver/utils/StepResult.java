package org.brain4j.math.solver.utils;

import org.brain4j.math.tensor.Tensor;

/**
 * Result of an adaptive ODE solver step, containing both the new state
 * and the recommended timestep for the next iteration.
 *
 * @param nextHidden the computed next hidden state
 * @param nextDeltaT the recommended timestep for the next step (may be adjusted based on error estimation)
 * @param accepted whether the step was accepted (error within tolerance) or rejected
 * @author Adversing
 */
public record StepResult(Tensor nextHidden, Tensor nextDeltaT, boolean accepted) {

    /**
     * Creates an accepted step result.
     *
     * @param nextHidden the computed next hidden state
     * @param nextDeltaT the recommended timestep for the next step
     * @return an accepted StepResult
     */
    public static StepResult accepted(Tensor nextHidden, Tensor nextDeltaT) {
        return new StepResult(nextHidden, nextDeltaT, true);
    }

    /**
     * Creates a rejected step result (error exceeded tolerance).
     *
     * @param currentHidden the current hidden state (unchanged)
     * @param reducedDeltaT the reduced timestep to retry with
     * @return a rejected StepResult
     */
    public static StepResult rejected(Tensor currentHidden, Tensor reducedDeltaT) {
        return new StepResult(currentHidden, reducedDeltaT, false);
    }
}

