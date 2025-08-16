package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

public class NormalUpdater extends Updater {
    
    public NormalUpdater(double learningRate) {
        super(learningRate);
    }
    
    @Override
    public void postFit(int samples) {
        updateWeights(samples);
        resetGradients();
    }
}