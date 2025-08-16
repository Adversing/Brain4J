package org.brain4j.common.data;

import org.brain4j.common.tensor.Tensor;

import java.util.Arrays;

public class Sample {
    private final Tensor[] inputs;
    private final Tensor[] labels;

    public Sample(Tensor input, Tensor label) {
        this(new Tensor[]{input}, new Tensor[]{label});
    }

    public Sample(Tensor[] inputs, Tensor[] labels) {
        this.inputs = inputs;
        this.labels = labels;
    }

    public Tensor[] inputs() {
        return inputs;
    }

    public Tensor input() {
        return inputs[0];
    }

    public Tensor[] labels() {
        return labels;
    }

    public Tensor label() {
        return labels[0];
    }

    @Override
    public String toString() {
        return Arrays.toString(inputs) + " -> " + Arrays.toString(labels);
    }
}
