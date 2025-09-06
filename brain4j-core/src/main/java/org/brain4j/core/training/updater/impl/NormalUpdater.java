package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

public class NormalUpdater extends Updater {
    
    @Override
    public void postFit(double learningRate, int samples) {
        updateWeights(learningRate, samples);
        resetGradients();
    }
}