package org.brain4j.math.tensor.broadcast.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.broadcast.BroadcastOperation;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BroadcastAdd implements BroadcastOperation {

    @Override
    public Tensor defaultOp(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();
        
        float[] aData = A.data();
        float[] bData = B.data();
        
        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0, n = aData.length; i < n; i++) {
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
                
                if (total > 65536) {
                    IntStream.range(0, d0).parallel().forEach(batch -> {
                        int offset = batch * stride;
                        for (int i = 0; i < stride; i++) {
                            aData[offset + i] += bData[i];
                        }
                    });
                } else {
                    for (int batch = 0; batch < d0; batch++) {
                        int offset = batch * stride;
                        for (int i = 0; i < stride; i++) {
                            aData[offset + i] += bData[i];
                        }
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

        int[] broadcastedShape = broadcastShape(shapeA, shapeB);

        if (!Arrays.equals(shapeA, broadcastedShape)) {
            throw new IllegalArgumentException("Broadcast result does not match shape of A");
        }

        int total = A.elements();

        int[] stridesB = B.strides();
        int[] index = new int[shapeA.length];

        for (int i = 0; i < total; i++) {
            unravelIndex(i, shapeA, index);

            int bIndex = 0;

            for (int d = 0; d < shapeA.length; d++) {
                int dimB = (shapeB.length - shapeA.length + d);
                
                if (dimB >= 0 && shapeB[dimB] != 1) {
                    bIndex += index[d] * stridesB[dimB];
                }
            }

            aData[i] += bData[bIndex];
        }

        return A;
    }
}
