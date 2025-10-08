package org.brain4j.math.data;

import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

public class Sample implements Cloneable {
    
    private Tensor[] inputs;
    private Tensor[] labels;

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
    
    @Override
    protected Sample clone() {
        try {
            Tensor[] clonedInputs = new Tensor[inputs.length];
            Tensor[] clonedLabels = new Tensor[labels.length];
            
            for (int i = 0; i < clonedInputs.length; i++) {
                clonedInputs[i] = inputs[i].clone();
            }
            
            for (int i = 0; i < clonedLabels.length; i++) {
                clonedLabels[i] = labels[i].clone();
            }
            
            Sample clone = (Sample) super.clone();
            
            clone.inputs = clonedInputs;
            clone.labels = clonedLabels;
            
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
