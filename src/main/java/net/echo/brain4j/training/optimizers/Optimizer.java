package net.echo.brain4j.training.optimizers;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

/**
 * Interface providing various methods to optimize training.
 */
@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer {

    protected double learningRate;

    /**
     * Initializes the optimizer with a specified learning rate.
     *
     * @param learningRate the learning rate
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param synapse the synapse to update
     */
    public abstract double update(StatesCache cache, Synapse synapse);

    public abstract double update(StatesCache cache, int id, float gradient, float weight);

    /**
     * Called after the network has been compiled and all the synapses have been initialized.
     */
    public void postInitialize(Sequential model) {
    }

    /**
     * Gets the current learning rate.
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets a new learning rate.
     *
     * @param learningRate the new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Called after a sample has been iterated.
     *
     * @param updater the backpropagation updater
     * @param layers the layers of the model
     */
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer<?, ?>> layers) {
    }
}