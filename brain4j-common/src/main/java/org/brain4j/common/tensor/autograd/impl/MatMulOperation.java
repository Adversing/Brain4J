package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class MatMulOperation implements Operation {
    
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].matmul(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        System.out.println("A: " + a);
        System.out.println("B: " + b);
        
        Tensor gradA = gradOutput.matmul(b.transpose());
        Tensor gradB = a.transpose().matmul(gradOutput);
        
        System.out.println("B hashcode: " + b.hashCode());
        System.out.println("Gradient for B: " + gradB);
        
        return new Tensor[] { gradA, gradB };
    }
} 