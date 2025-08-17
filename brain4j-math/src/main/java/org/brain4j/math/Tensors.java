package org.brain4j.math;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.parallel.ParallelConvolve;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;

public class Tensors {

    public static Tensor scalar(double value) {
        return new CpuTensor(new int[]{1}, (float) value);
    }

    public static Tensor create(int[] shape, float... data) {
        return new CpuTensor(shape, data);
    }

    public static Tensor create(int[] shape, int[] strides, float[] data) {
        return new CpuTensor(shape, strides, data);
    }

    public static Tensor vector(float... data) {
        return create(new int[]{data.length}, data);
    }

    public static Tensor matrix(int rows, int cols, float... data) {
        return create(new int[]{rows, cols}, data);
    }

    public static Tensor zeros(int... shape) {
        return new CpuTensor(shape);
    }

    public static Tensor range(int start, int end) {
        int length = end - start;
        Tensor result = Tensors.zeros(length);

        for (int i = 0; i < length; i++) {
            result.data()[i] = i + start;
        }

        return result;
    }

    public static Tensor ones(int... shape) {
        Tensor result = new CpuTensor(shape);
        Arrays.fill(result.data(), 1);
        return result;
    }

    public static Tensor random(Random generator, int... shape) {
        Tensor result = Tensors.zeros(shape);
        return result.map(x -> generator.nextFloat());
    }

    public static Tensor random(int... shape) {
        return random(Random.from(new SplittableRandom()), shape);
    }

    public static Tensor convolve(Tensor input, Tensor kernel) {
        return ParallelConvolve.convolve(input, kernel);
    }
    
    public static Tensor mergeTensors(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("No tensors provided!");
        }

        Tensor first = tensors.getFirst();
        int dimension = first.rank();

        int[] shape = first.shape();
        int[] newShape = new int[dimension + 1];

        newShape[0] = tensors.size();
        System.arraycopy(shape, 0, newShape, 1, dimension);

        Tensor result = zeros(newShape);

        for (int i = 0; i < tensors.size(); i++) {
            Tensor current = tensors.get(i);

            if (current.rank() != dimension) {
                throw new IllegalArgumentException(
                        "All input tensors must have the same dimension!"
                );
            }

            int[] idx = new int[dimension];
            copyRecursive(current, result, idx, 0, i);
        }

        return result;
    }

    private static void copyRecursive(Tensor src, Tensor dest, int[] idx, int dim, int batchIndex) {
        if (dim == idx.length) {
            float value = src.get(idx);

            int[] destIdx = new int[idx.length + 1];
            destIdx[0] = batchIndex;

            System.arraycopy(idx, 0, destIdx, 1, idx.length);

            dest.set(value, destIdx);
        } else {
            int dimSize = src.shape()[dim];

            for (int j = 0; j < dimSize; j++) {
                idx[dim] = j;
                copyRecursive(src, dest, idx, dim + 1, batchIndex);
            }
        }
    }

    public static Tensor triangularMask(int dimension) {
        Tensor mask = Tensors.zeros(dimension, dimension);

        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                mask.set(Float.NEGATIVE_INFINITY, i, j);
            }
        }

        return mask;
    }

    public static Tensor concat(List<Tensor> tensors) {
        return concat(tensors, -1);
    }

    public static Tensor concat(List<Tensor> tensors, int dim) {
        Tensor base = tensors.getFirst();

        for (int i = 1; i < tensors.size(); i++) {
            base = base.concat(tensors.get(i), dim);
        }

        return base;
    }

    public static Tensor concatGrad(List<Tensor> tensors) {
        return concatGrad(tensors, -1);
    }

    public static Tensor concatGrad(List<Tensor> tensors, int dim) {
        Tensor base = tensors.getFirst();
        
        for (int i = 1; i < tensors.size(); i++) {
            base = base.concatGrad(tensors.get(i), dim);
        }
        
        return base;
    }

    public static int[] broadcastShapes(int[] a, int[] b) {
        int len = Math.max(a.length, b.length);
        int[] result = new int[len];
        for (int i = 0; i < len; i++) {
            int ai = i >= len - a.length ? a[i - (len - a.length)] : 1;
            int bi = i >= len - b.length ? b[i - (len - b.length)] : 1;
            if (ai != bi && ai != 1 && bi != 1)
                throw new IllegalArgumentException("Incompatible dimensions for broadcasting");
            result[i] = Math.max(ai, bi);
        }
        return result;
    }

    public static int[] broadcastIndex(int[] outIdx, int[] outShape, int[] targetShape) {
        int offset = outShape.length - targetShape.length;
        int[] result = new int[targetShape.length];

        for (int i = 0; i < targetShape.length; i++) {
            result[i] = targetShape[i] == 1 ? 0 : outIdx[i + offset];
        }

        return result;
    }
    
    public static int flattenIndex(int[] idx, int[] strides) {
        int sum = 0;

        for (int i = 0; i < idx.length; i++) {
            sum += idx[i] * strides[i];
        }

        return sum;
    }
    
    public static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int prod = 1;

        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = prod;
            prod *= shape[i];
        }

        return strides;
    }
    
    public static int[] unravelIndex(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
        }

        return indices;
    }
    
    public static void validateShape(Tensor a, Tensor b) {
        int[] shapeA = a.shape();
        int[] shapeB = b.shape();

        if (shapeA.length != shapeB.length) {
            throw new IllegalArgumentException("Tensors dimensions must match: " + shapeA.length + " != " + shapeB.length);
        }

        for (int i = 0; i < shapeA.length; i++) {
            if (shapeA[i] != shapeB[i]) {
                throw new IllegalArgumentException(
                    "Tensors shapes must match: " + Arrays.toString(shapeA) + " != " + Arrays.toString(shapeB)
                );
            }
        }
    }

    public static int computeSize(int[] shape) {
        int size = 1;

        for (int dim : shape) {
            size *= dim;
        }

        return size;
    }

    public static int[] computeNewShape(int[] shape, int dim, boolean keepDim) {
        int[] newShape = keepDim ? Arrays.copyOf(shape, shape.length) : new int[shape.length - 1];

        if (keepDim) {
            newShape[dim] = 1;
        } else {
            for (int i = 0, j = 0; i < shape.length; i++) {
                if (i != dim) {
                    newShape[j++] = shape[i];
                }
            }
        }

        return newShape;
    }
}