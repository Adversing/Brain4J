package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.NormalXavierInit;
import org.brain4j.math.weightsinit.WeightInitialization;

public class SwishActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalXavierInit();
    }

    @Override
    public double activate(double input) {
        return input * (1.0 / (1.0 + Math.exp(-input)));
    }

    @Override
    public double derivative(double input) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));
        return sigmoid + input * sigmoid * (1 - sigmoid);
    }

    @Override
    public String kernelPrefix() {
        return "swish";
    }
}
