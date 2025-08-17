package org.brain4j.common.tensor.matmul;

import org.brain4j.common.tensor.Tensor;

public interface MatmulProvider {

    void multiply(Tensor a, Tensor b, Tensor c);
}
