package org.brain4j.math.weightsinit;

import java.util.random.RandomGenerator;

public class NormalXavierInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 2.0 / (input + output);
    }

    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, 0, getBound(input, output));
    }
}