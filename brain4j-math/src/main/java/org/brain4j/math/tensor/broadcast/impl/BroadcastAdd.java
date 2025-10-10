package org.brain4j.math.tensor.broadcast.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.broadcast.BroadcastOperation;

import java.util.Arrays;

public class BroadcastAdd implements BroadcastOperation {

    @Override
    public Tensor defaultOp(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();
        
        float[] aData = A.data();
        float[] bData = B.data();
        
        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] += bData[i];
            }
            
            return A;
        }
        
        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];
            
            int total = batch * dimension;
            for (int idx = 0; idx < total; idx++) {
                int j = idx % dimension;
                aData[idx] += bData[j];
            }
            
            return A;
        }

        if (shape.length == 3) {
            int d0 = shape[0]; // a
            int d1 = shape[1]; // b
            int d2 = shape[2]; // c

            int total = d0 * d1 * d2;

            // [a, b, c] + [c]
            if (otherShape.length == 1 && shape[2] == otherShape[0]) {
                for (int idx = 0; idx < total; idx++) {
                    int k = idx % d2;
                    aData[idx] += bData[k];
                }

                return A;
            }

            // [a, b, c] + [b, c]
            if (otherShape.length == 2 && shape[1] == otherShape[0] && shape[2] == otherShape[1]) {
                int stride = d1 * d2; // size of [b, c]
                
                for (int batch = 0; batch < d0; batch++) {
                    int offset = batch * stride;
                    for (int i = 0; i < stride; i++) {
                        aData[offset + i] += bData[i];
                    }
                }

                return A;
            }
        }

        return fallbackOp(A, B);
    }

    @Override
    public Tensor fallbackOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();
        float[] aData = A.data();
        float[] bData = B.data();

        int ndimA = shapeA.length;
        int ndimB = shapeB.length;
        int[] stridesB = B.strides();

        int total = A.elements();
        int[] indexA = new int[ndimA];

        for (int i = 0; i < total; i++) {
            int bIndex = 0;
            int dimOffset = ndimB - ndimA;

            for (int d = 0; d < ndimA; d++) {
                int dimB = dimOffset + d;
                if (dimB >= 0) {
                    int sizeB = shapeB[dimB];
                    int idx = (sizeB == 1) ? 0 : indexA[d];
                    bIndex += idx * stridesB[dimB];
                }
            }

            aData[i] += bData[bIndex];

            for (int d = ndimA - 1; d >= 0; d--) {
                if (++indexA[d] < shapeA[d]) break;
                indexA[d] = 0;
            }
        }

        return A;
    }
}
