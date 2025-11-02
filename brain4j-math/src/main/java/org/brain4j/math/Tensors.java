package org.brain4j.math;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.convolve.Im2ColParams;
import org.brain4j.math.tensor.convolve.Im2ColTask;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.tensor.parallel.ParallelConvolve;
import org.brain4j.math.tensor.parallel.ParallelTranspose;

import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * Utility factory and helper methods for creating and manipulating {@link Tensor} instances.
 *
 * <p>This class centralizes frequently used tensor constructors (scalars, vectors, matrices,
 * zeros/ones), random generation helpers, broadcasting logic, and a number of indexing
 * and shape utilities used across the library.
 *
 * <p>The helpers are implemented for CPU-backed tensors and delegate to the underlying
 * tensor implementations (for example {@link org.brain4j.math.tensor.impl.CpuTensor}).
 */
public class Tensors {

    /** Number of threads suggested for parallel operations. */
    public static final int PARALLELISM = Runtime.getRuntime().availableProcessors();

    /** Complexity threshold used to decide when to split tasks for parallel algorithms. */
    public static final int SPLIT_COMPLEXITY_THRESHOLD = 1 << 10; // 1024

    public static Tensor scalar(double value) {
        return new CpuTensor(new int[]{1}, (float) value);
    }

    /**
     * Create a scalar {@link Tensor} containing the supplied value.
     *
     * @param value the scalar value
     * @return a rank-1 tensor of length 1 containing the value
     */

    public static Tensor create(int[] shape, float... data) {
        return new CpuTensor(shape, data);
    }

    /**
     * Create a {@link Tensor} with explicit strides and raw data buffer.
     *
     * @param shape the tensor shape
     * @param strides explicit strides for the tensor
     * @param data raw float data in row-major order according to given strides
     * @return a new tensor instance
     */

    public static Tensor create(int[] shape, int[] strides, float... data) {
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

    /**
     * Returns a tensor filled with zeros of the requested shape.
     *
     * @param shape desired shape
     * @return a tensor initialized with zeros
     */

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

    public static Tensor random(RandomGenerator generator, int... shape) {
        Tensor result = Tensors.zeros(shape);
        return result.map(x -> generator.nextFloat());
    }

    public static Tensor random(int... shape) {
        return random(new SplittableRandom(), shape);
    }

    public static Tensor convolve(Tensor input, Tensor kernel) {
        return ParallelConvolve.convolve(input, kernel);
    }
    
    public static Tensor mergeTensors(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("No tensors provided!");
        }

        Tensor result = tensors.getFirst().unsqueeze();

        for (int i = 1; i < tensors.size(); i++) {
            Tensor current = tensors.get(i).unsqueeze();
            result = result.concat(current, 0);
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

    public static Tensor triangularMask(int firstDim, int secondDim) {
        Tensor mask = Tensors.zeros(firstDim, secondDim);

        for (int i = 0; i < firstDim; i++) {
            for (int j = i + 1; j < secondDim; j++) {
                mask.set(Float.NEGATIVE_INFINITY, i, j);
            }
        }

        return mask;
    }

    public static Tensor triangularMask(int dimension) {
        return triangularMask(dimension, dimension);
    }

    /**
     * Create a triangular mask tensor of shape (dimension, dimension) with
     * negative infinity above the diagonal and zeros on and below it. Useful
     * for attention masking in transformer models.
     *
     * @param dimension the square mask dimension
     * @return a triangular mask tensor
     */

    public static Tensor concat(Tensor... tensors) {
        return concat(List.of(tensors), -1);
    }

    /**
     * Concatenate tensors along the provided dimension. If dim is -1 the last
     * dimension is used.
     */

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

    /**
     * Compute the broadcasted shape of two shapes according to NumPy-style broadcasting rules.
     *
     * @param a first shape
     * @param b second shape
     * @return resulting broadcast shape
     */

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

    public static Tensor orthogonal(int rows, int cols) {
        Random rng = new Random();
        Tensor A = Tensors.zeros(rows, cols).map(x -> rng.nextGaussian());

        List<Tensor> Q = new ArrayList<>();

        for (int i = 0; i < cols; i++) {
            Tensor ai = A.slice(Range.all(), Range.point(i));
            Tensor ui = ai;

            for (Tensor qj : Q) {
                Tensor dot = qj.transpose().matmul(ai); // [1, 1]
                ui = ui.minus(qj.times(dot.get(0, 0)));
            }

            double norm = Math.sqrt(ui.clone().pow(2).sum());
            Tensor qi = ui.divide(norm);

            Q.add(qi);
        }

        return Tensors.concat(Q, 1);
    }
    
    public static Tensor eye(int n) {
        Tensor result = Tensors.zeros(n, n);

        for (int i = 0; i < n; i++) {
            result.set(1, i, i);
        }

        return result;
    }

    public static Tensor zerosLike(Tensor a) {
        return Tensors.zeros(a.shape());
    }

    public static Tensor im2col(Tensor input, int filterHeight, int filterWidth) {
        int[] shape = input.shape();
        int channels = shape[0];
        int inHeight = shape[1];
        int inWidth = shape[2];

        int outHeight = inHeight - filterHeight + 1;
        int outWidth = inWidth - filterWidth + 1;

        int patchSize = channels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        float[] inputData = input.data();
        float[] resultData = new float[patchSize * totalPatches];

        Im2ColParams params = new Im2ColParams(
            inputData,
            resultData,
            0,
            0,
            channels,
            inHeight,
            inWidth,
            filterHeight,
            filterWidth,
            outHeight,
            outWidth
        );

        try (var pool = ForkJoinPool.commonPool()) {
            pool.invoke(new Im2ColTask(params, 0, totalPatches));
        }

        return Tensors.create(new int[]{patchSize, totalPatches}, resultData);
    }

    public static Tensor col2im(Tensor cols, int channels, int inHeight, int inWidth,
                                int filterHeight, int filterWidth) {
        int[] colShape = cols.shape(); // [patchSize, totalPatches]
        int patchSize = colShape[0];
        int totalPatches = colShape[1];

        int outWidth = inWidth - filterWidth + 1;

        float[] colData = cols.data();
        float[] imgData = new float[channels * inHeight * inWidth];

        for (int patchIdx = 0; patchIdx < totalPatches; patchIdx++) {
            int outRow = patchIdx / outWidth;
            int outCol = patchIdx % outWidth;
            int baseOffset = patchIdx * patchSize;

            for (int c = 0; c < channels; c++) {
                int channelOffset = c * filterHeight * filterWidth;
                int imgChannelOffset = c * inHeight * inWidth;

                for (int fh = 0; fh < filterHeight; fh++) {
                    int srcPos = baseOffset + channelOffset + fh * filterWidth;
                    int destPos = imgChannelOffset + (outRow + fh) * inWidth + outCol;
                    for (int fw = 0; fw < filterWidth; fw++) {
                        imgData[destPos + fw] += colData[srcPos + fw];
                    }
                }
            }
        }

        return Tensors.create(new int[]{channels, inHeight, inWidth}, imgData);
    }

    public static int[] topK(int topK, float[] data) {
       return IntStream.range(0, data.length)
            .boxed()
            .sorted((i, j) -> Double.compare(data[j], data[i]))
            .limit(topK)
            .mapToInt(Integer::intValue)
            .toArray();
    }
}