package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;
import org.brain4j.common.tensor.index.Range;

import java.util.Arrays;

public class ConcatOperation implements Operation {

    private final int dimension;

    public ConcatOperation(int dimension) {
        this.dimension = dimension;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        if (dimension == -1) {
            return inputs[0].concat(inputs[1]);
        }

        return inputs[0].concat(inputs[1], dimension);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        if (inputs.length != 2) {
            throw new IllegalArgumentException("ConcatOperation supports exactly two input tensors.");
        }

        Tensor a = inputs[0];
        Tensor b = inputs[1];

        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int rank = shapeA.length;

        if (dimension < 0 || dimension >= rank) {
            throw new IllegalArgumentException("Invalid concat dimension: " + dimension);
        }

        int sizeA = shapeA[dimension];
        int sizeB = shapeB[dimension];

        // Costruisci il range "base"
        Range[] base = new Range[rank];
        for (int i = 0; i < rank; i++) {
            base[i] = Range.all();
        }

        // Slice per A: dimensione da 0 a sizeA
        Range[] rangeA = base.clone();
        rangeA[dimension] = new Range(0, sizeA);

        // Slice per B: dimensione da sizeA a sizeA + sizeB
        Range[] rangeB = base.clone();
        rangeB[dimension] = new Range(sizeA, sizeA + sizeB);

        Tensor gradA = gradOutput.slice(rangeA);
        Tensor gradB = gradOutput.slice(rangeB);

        return new Tensor[]{ gradA, gradB };
    }
}
