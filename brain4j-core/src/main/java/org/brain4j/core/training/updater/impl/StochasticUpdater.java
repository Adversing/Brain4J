package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

public class StochasticUpdater extends Updater {
    
    @Override
    public void postBatch(double learningRate, int samples) {
        updateWeights(learningRate, samples);
        resetGradients();
    }
}