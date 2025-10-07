package org.brain4j.math.weightsinit;

import java.util.random.RandomGenerator;

public class UniformHeInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return Math.sqrt(6.0 / input);
    }

    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        double bound = getBound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}