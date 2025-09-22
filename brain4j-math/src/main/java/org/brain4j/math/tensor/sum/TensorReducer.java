package org.brain4j.math.tensor.sum;

import org.brain4j.math.tensor.Tensor;

public interface TensorReducer {
    Tensor sum(Tensor tensor, int dim, boolean keepDim);
}
