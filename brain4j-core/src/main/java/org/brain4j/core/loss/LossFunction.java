package org.brain4j.core.loss;

import org.brain4j.math.tensor.Tensor;

/**
 * Loss functions (also called cost functions) are used during training
 * and to measure the performance of a network.
 * @author xEcho1337
 */
public interface LossFunction {
    /**
     * Calculates the error of the network given the expected and predicted tensor.
     * @param expected the expected output
     * @param predicted the predicted output
     * @return a scalar value
     */
    double calculate(Tensor expected, Tensor predicted);

    /**
     * Calculates the delta for the last org.brain4j.core.layer of the network.
     * @param error usually calculated as <code>expected - predicted</code>
     * @param derivative the derivative of the activation function
     * @return the delta tensor
     */
    Tensor delta(Tensor error, Tensor derivative);

    /**
     * Gets whether this loss function is typically used for regression.
     * @return true if this is used for regression, false otherwise
     */
    boolean isRegression();
}
