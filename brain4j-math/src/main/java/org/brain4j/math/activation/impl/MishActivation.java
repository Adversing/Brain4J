package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.NormalXavierInit;
import org.brain4j.math.weightsinit.WeightInitialization;

public class MishActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalXavierInit();
    }

    @Override
    public double activate(double input) {
        double softplus = Math.log1p(Math.exp(input));
        return input * Math.tanh(softplus);
    }

    @Override
    public double derivative(double input) {
        double softplus = Math.log1p(Math.exp(input));
        double tanhSp = Math.tanh(softplus);
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));

        return tanhSp + input * sigmoid * (1 - tanhSp * tanhSp);
    }

    @Override
    public String kernelPrefix() {
        return "mish";
    }
}
