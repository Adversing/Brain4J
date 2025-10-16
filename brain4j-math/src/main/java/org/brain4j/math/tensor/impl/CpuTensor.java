package org.brain4j.math.tensor.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.broadcast.TensorBroadcast;
import org.brain4j.math.tensor.matmul.MatmulProvider;
import org.brain4j.math.tensor.matmul.impl.NormalMatmulProvider;
import org.brain4j.math.tensor.matmul.impl.SimdMatmulProvider;
import org.brain4j.math.tensor.parallel.ParallelTranspose;

import java.util.Arrays;

public class CpuTensor extends BaseTensor {

    private static final MatmulProvider matmulProvider;

    static {
        if (DeviceUtils.isSimdAvailable()) {
            matmulProvider = new SimdMatmulProvider();
        } else {
            System.err.println("The Vector incubator API is not available. It's recommended to use for better performance.");
            System.err.println("For more information consult this guide: https://github.com/brain4j-org/brain4j/wiki/Using-SIMD");

            matmulProvider = new NormalMatmulProvider();
        }
    }

    public CpuTensor(int[] shape, float... data) {
        if (data.length == 0) {
            data = new float[Tensors.computeSize(shape)];
        }

        this.data = data;
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);
    }

    public CpuTensor(int[] shape, int[] strides, float... data) {

        if (data.length == 0) {
            data = new float[Tensors.computeSize(shape)];
        }

        this.data = data;
        this.shape = shape;
        this.strides = strides;
    }
    
    @Override
    public Tensor transpose() {
        int rank = shape.length;
        return transpose(rank - 2, rank - 1);
    }

    @Override
    public Tensor transpose(int dim1, int dim2) {
        // Unfortunately, SIMD does not support non-contiguous data, therefore transposing
        // the data in a contiguous space is required for SIMD matmul to work
        if (matmulProvider instanceof NormalMatmulProvider) {
            return super.transpose(dim1, dim2);
        }

        int rank = shape.length;

        if (rank == 1) {
            return reshape(1, elements());
        }

        int[] newShape = shape.clone();

        int rows = shape[dim1];
        int cols = shape[dim2];

        newShape[dim1] = cols;
        newShape[dim2] = rows;

        BaseTensor result = (BaseTensor) Tensors.create(newShape);

        int bound = 1 << 10;

        // TODO: fix parallel transpose?
        if (elements() >= bound) {
            ParallelTranspose.transpose(this, result, dim1, dim2);
            return result;
        }

        int[] srcStride = this.strides;
        int[] dstStride = result.strides;
        int[] loopShape = result.shape;

        int[] destToSrc = new int[rank];
        for (int d = 0; d < rank; d++) destToSrc[d] = d;

        destToSrc[dim1] = dim2;
        destToSrc[dim2] = dim1;

        transposeRecursive(
            this.data, result.data,
            loopShape, srcStride, dstStride, destToSrc,
            0, 0, 0
        );

        return result;
    }

    private void transposeRecursive(
        float[] src, float[] dst,
        int[] loopShape,
        int[] srcStride, int[] dstStride,
        int[] destToSrc,
        int dim, int srcOffset, int dstOffset
    ) {
        if (dim == loopShape.length) {
            dst[dstOffset] = src[srcOffset];
            return;
        }

        int sStride = srcStride[destToSrc[dim]];
        int dStride = dstStride[dim];
        int extent = loopShape[dim];

        int s = srcOffset;
        int d = dstOffset;
        for (int i = 0; i < extent; i++) {
            transposeRecursive(src, dst, loopShape, srcStride, dstStride, destToSrc,
                dim + 1, s, d);
            s += sStride;
            d += dStride;
        }
    }

    @Override
    public Tensor to(Device device) {
        if (device == null) {
            return this;
        }

        GpuTensor result = new GpuTensor(device, shape, data);
        result.setAutogradContext(autogradContext);
        return result;
    }

    @Override
    public Tensor add(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return add(other.cpu());
        }

        return TensorBroadcast.add(this, other);
    }
    
    @Override
    public Tensor add(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] += (float) value;
        }
        
        return this;
    }
    
    @Override
    public Tensor sub(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return sub(other.cpu());
        }

        return TensorBroadcast.sub(this, other);
    }
    
    @Override
    public Tensor sub(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] -= (float) value;
        }
        
        return this;
    }
    
    @Override
    public Tensor mul(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return mul(other.cpu());
        }

        return TensorBroadcast.mul(this, other);
    }
    
    @Override
    public Tensor mul(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] *= (float) value;
        }
        
        return this;
    }

    @Override
    public Tensor div(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return div(other.cpu());
        }

        return TensorBroadcast.div(this, other);
    }
    
    @Override
    public Tensor div(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] /= (float) value;
        }
        
        return this;
    }
    
    @Override
    public Tensor pow(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return pow(other.cpu());
        }
        
        return TensorBroadcast.pow(this, other);
    }
    
    @Override
    public Tensor pow(double value) {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.pow(data[i], value);
        }
        
        return this;
    }
    
    @Override
    public Tensor sqrt() {
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) Math.sqrt(data[i]);
        }
        
        return this;
    }
    
    @Override
    public Tensor matmul(Tensor other) {
        int[] shapeA = this.shape;
        int[] shapeB = other.shape();

        if (shapeA.length < 2 || shapeB.length < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires at least 2D tensors!");
        }

        int rankA = shapeA.length;
        int rankB = shapeB.length;

        int m = shapeA[rankA - 2];
        int n = shapeA[rankA - 1];

        int k = shapeB[rankB - 2];
        int p = shapeB[rankB - 1];

        if (n != k) {
            throw new IllegalArgumentException("Inner dimensions must match: " + n + " != " + k +
                ". A: " + Arrays.toString(shapeA) + ", B: " + Arrays.toString(shapeB));
        }

        int maxBatchDims = Math.max(rankA, rankB) - 2;
        int[] batchShape = new int[maxBatchDims];

        for (int i = 0; i < maxBatchDims; i++) {
            int dimA = (i < rankA - 2) ? shapeA[i + rankA - 2 - maxBatchDims] : 1;
            int dimB = (i < rankB - 2) ? shapeB[i + rankB - 2 - maxBatchDims] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException(
                    "Cannot broadcast batch dimensions: " + dimA + " vs " + dimB + " at batch dim index " + i
                );
            }

            batchShape[i] = Math.max(dimA, dimB);
        }

        int[] resultShape = new int[batchShape.length + 2];

        System.arraycopy(batchShape, 0, resultShape, 0, batchShape.length);

        resultShape[resultShape.length - 2] = m;
        resultShape[resultShape.length - 1] = p;

        Tensor result = new CpuTensor(resultShape);

        matmulProvider.multiply(this, other, result);

        return result;
    }
}
