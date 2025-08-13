package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

import java.util.Arrays;

public class MatMulOperation implements Operation {
    
    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
//        System.out.println("Matmul of " + Arrays.toString(a.shape()) + " with " + Arrays.toString(b.shape()));
        
        return a.matmul(b);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        Tensor aT = a.transpose();
        Tensor bT = b.transpose();
        
        System.out.println("A shape: " + Arrays.toString(aT.shape()));
        System.out.println("Gradout shape: " + Arrays.toString(gradOutput.shape()));
        
        Tensor gradA = gradOutput.matmul(bT);
        Tensor gradB = aT.matmul(gradOutput);
        
        return new Tensor[] { gradA, gradB };
    }
} 