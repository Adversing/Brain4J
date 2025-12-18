package org.brain4j.math.commons;

import org.brain4j.math.tensor.Tensor;

public class Batch extends Pair<Tensor[], Tensor[]> {

    public Batch(Tensor[] key, Tensor[] second) {
        super(key, second);
    }
}
