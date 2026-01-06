package org.brain4j.math.tensor;

import org.brain4j.math.commons.Commons;

public class Shape {

    private final int[] dims;

    private Shape(int[] dims) {
        for (int dim : dims) {
            if (dim < 0) {
                throw Commons.illegalArgument("Dimension at %s is negative!", dim);
            }
        }

        this.dims = dims.clone();
    }

    public static Shape of(int... dims) {
        return new Shape(dims);
    }

    public int dim(int index) {
        return dims[index];
    }

    public int last() {
        return dims[dims.length - 1];
    }

    public int[] dims() {
        return dims;
    }
}
