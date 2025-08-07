package org.brain4j.core.model;

import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.core.training.wrappers.TrainingParams;

import java.util.List;

/**
 * Represents a generic neural network.
 *
 * @author xEcho1337
 * @since 3.0
 */
public interface Model extends Iterable<Layer> {
    
    static Model of(Layer... layers) {
        throw new UnsupportedOperationException("Static factory method must be implemented in a concrete class.");
    }
    
    /**
     * Adds a layer at the end of the network.
     * @param layer the layer to add
     * @return this instance
     */
    Model add(Layer layer);

    /**
     * Adds a layer at the specified position.
     * @param index the position to insert the layer
     * @param layer the layer to insert
     * @return this instance
     */
    Model add(int index, Layer layer);

    /**
     * Predicts the output from the input tensor.
     * @param inputs the input tensors
     * @return the output tensor
     */
    default Tensor predict(Tensor... inputs) {
        return predict(new StatesCache(device()), inputs);
    }

    /**
     * Predicts output using a cache and input tensor.
     * @param cache the states cache
     * @param inputs the input tensors
     * @return the output tensor
     */
    default Tensor predict(StatesCache cache, Tensor... inputs) {
        return predict(cache, false, inputs);
    }

    /**
     * Predicts output with optional training mode.
     * @param cache the states cache
     * @param inputs the input tensors
     * @param training whether in training mode
     * @return the output tensor
     */
    Tensor predict(StatesCache cache, boolean training, Tensor... inputs);

    /**
     * Executes a backpropagation pass for this model
     * @param cache the states cache
     * @param outputs the outputs of the network
     * @param targets the target outputs
     */
    void backpropagate(StatesCache cache, Tensor outputs, Tensor targets);

    /**
     * Trains the model using full training parameters.
     * @param params the training parameters
     */
    default void fit(TrainingParams params) {
        fit(params.train(), params.validation(), params.epochs(), params.evaluateEvery());
    }

    /**
     * Trains the model on a dataset.
     * @param train the training dataset
     */
    default void fit(ListDataSource train) {
        fit(train, train, 1, Integer.MAX_VALUE);
    }

    /**
     * Trains the model with training and validation datasets.
     * @param train training dataset
     * @param validation validation dataset
     */
    default void fit(ListDataSource train, ListDataSource validation) {
        fit(train, validation, 1, Integer.MAX_VALUE);
    }

    /**
     * Trains the model for a number of epochs.
     * @param train training dataset
     * @param epoches number of epochs
     */
    default void fit(ListDataSource train, int epoches) {
        fit(train, train, epoches, Integer.MAX_VALUE);
    }

    /**
     * Trains the model with validation and epochs.
     * @param train training dataset
     * @param validation validation dataset
     * @param epoches number of epochs
     */
    default void fit(ListDataSource train, ListDataSource validation, int epoches) {
        fit(train, validation, epoches, Integer.MAX_VALUE);
    }

    /**
     * Trains the model with periodic evaluation.
     * @param train training dataset
     * @param epoches number of epochs
     * @param evaluateEvery evaluation frequency
     */
    default void fit(ListDataSource train, int epoches, int evaluateEvery) {
        fit(train, train, epoches, evaluateEvery);
    }

    /**
     * Trains the model with validation and evaluation frequency.
     * @param train training dataset
     * @param validation validation dataset
     * @param epoches number of epochs
     * @param evaluateEvery frequency for evaluation
     */
    void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery);

    /**
     * Evaluates the model on a dataset.
     * @param dataSource dataset to evaluate
     * @return result of evaluation
     */
    EvaluationResult evaluate(ListDataSource dataSource);

    /**
     * Computes average loss on the dataset.
     * @param dataSource dataset to compute loss on
     * @return average loss
     */
    double loss(ListDataSource dataSource);

    /**
     * Compiles the model by setting the loss function, optimizer, and default updater.
     *
     * @param lossFunction the loss function to use
     * @param optimizer the optimization algorithm
     * @return the compiled model instance for method chaining
     */
    default Model compile(LossFunction lossFunction, Optimizer optimizer) {
        return compile(lossFunction, optimizer, new StochasticUpdater());
    }

    /**
     * Compiles the model by setting the loss function, optimizer, and custom updater.
     *
     * @param lossFunction the loss function to use
     * @param optimizer the optimization algorithm
     * @param updater the updater managing gradient application
     * @return the compiled model instance for method chaining
     */
    Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater);

    /**
     * Moves the model weights to the specified device.
     * @param device the device
     * @return the current model instance
     */
    Model to(Device device);

    /**
     * Gets the device the model is stored on.
     * @return the device type of this model
     */
    Device device();

    /**
     * Returns an immutable list of layers composing the model.
     * @return the list of layers
     */
    List<Layer> layers();

    /**
     * Returns an immutable list of all the layers in the model, including nested layers.
     * @return the list of layers
     */
    List<Layer> flattened();

    /**
     * Returns the layer at the specified index.
     * @param index the index
     * @return the layer in that index
     */
    Layer layerAt(int index);

    /**
     * Returns the flattened layer at the specified index.
     * @param index the index
     * @return the layer in that index
     */
    Layer flattenedAt(int index);

    /**
     * Returns the optimizer currently used by the model.
     * @return the optimizer instance
     */
    default Optimizer optimizer() {
        return null;
    }

    default void setOptimizer(Optimizer optimizer) {
    }
    
    /**
     * Returns the updater currently used by the model.
     * @return the updater instance
     */
    default Updater updater() {
        return null;
    }
    
    default void setUpdater(Updater updater) {
    }
    
    /**
     * Returns the loss function currently set in the model.
     * @return the loss function instance
     */
    default LossFunction lossFunction() {
        return null;
    }
    
    default void setLossFunction(LossFunction lossFunction) {
    }

    /**
     * Prints a formatted summary of the model architecture to the console,
     * including weights, dimensions and activations of the layers, along with the total parameters.
     *
     * @throws IllegalStateException if the model is not compiled before calling this method
     */
    void summary();

    /**
     * Resets all the gradients in the model.
     */
    void zeroGrad();
}
