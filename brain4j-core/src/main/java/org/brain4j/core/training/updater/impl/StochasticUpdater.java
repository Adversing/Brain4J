package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

public class StochasticUpdater extends Updater {
    
    public StochasticUpdater(double learningRate) {
        super(learningRate);
    }
    
    @Override
    public void postBatch(int samples) {
        updateWeights(samples);
        resetGradients();
    }
}