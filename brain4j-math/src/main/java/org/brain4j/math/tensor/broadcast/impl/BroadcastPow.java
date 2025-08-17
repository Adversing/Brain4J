package org.brain4j.math.tensor.broadcast.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.broadcast.BroadcastOperation;

import java.util.Arrays;

public class BroadcastPow implements BroadcastOperation {

    @Override
    public Tensor defaultOp(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] = (float) Math.pow(aData[i], bData[i]);
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] = (float) Math.pow(aData[base + j], bData[j]);
                }
            }

            return A;
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
        
        if (!Arrays.equals(shapeB, broadcastedShape)) {
            B = B.reshape(broadcastedShape);
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

            aData[i] = (float) Math.pow(aData[i], bData[bIndex]);
        }

        return A;
    }
}
