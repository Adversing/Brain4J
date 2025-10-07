package org.brain4j.math.weightsinit;

import java.util.random.RandomGenerator;

public class NormalInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 1;
    }

    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, -1, 1);
    }
}