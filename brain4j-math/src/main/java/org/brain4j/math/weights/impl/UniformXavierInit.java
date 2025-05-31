package org.brain4j.math.weights.impl;

import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

public class UniformXavierInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return Math.sqrt(6.0 / (input + output));
    }

    @Override
    public double generate(Random generator, int input, int output) {
        double bound = getBound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}