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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CpuTensor extends BaseTensor {

    private static final Logger LOGGER = LoggerFactory.getLogger(CpuTensor.class);
    private static final MatmulProvider matmulProvider;

    static {
        if (DeviceUtils.isSimdAvailable()) {
            matmulProvider = new SimdMatmulProvider();
        } else {
            LOGGER.warn("The Vector incubator API is not available. It's recommended to use for better performance.");
            LOGGER.warn("For more information consult this guide: https://github.com/brain4j-org/brain4j/wiki/Using-SIMD");

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
        // Unfortunately, SIMD does not support non-contiguous data, therefore transposing
        // the data in a contiguous space is required for SIMD matmul to work
        if (matmulProvider instanceof NormalMatmulProvider) {
            return super.transpose();
        }

        int rank = shape.length;

        if (rank == 1) {
            return reshape(1, elements());
        }

        int[] newShape = shape.clone();

        int rows = shape[rank - 2];
        int cols = shape[rank - 1];

        newShape[rank - 2] = cols;
        newShape[rank - 1] = rows;

        BaseTensor result = (BaseTensor) Tensors.create(newShape);

        int bound = 1 << 16;

        if (elements() >= bound) {
            ParallelTranspose.transpose(this, result);
            return result;
        }

        float[] source = this.data;
        float[] dest = result.data;

        int batch = 1;

        for (int i = 0; i < rank-2; i++) batch *= shape[i];

        int planeSize = rows * cols;

        for (int b = 0; b < batch; b++) {
            int srcOffset = b * planeSize;
            int dstOffset = b * planeSize;

            for (int i = 0; i < rows; i++) {
                int rowStart = srcOffset + i * cols;

                for (int j = 0; j < cols; j++) {
                    int srcIndex = rowStart + j;
                    int dstIndex = dstOffset + j * rows + i;

                    dest[dstIndex] = source[srcIndex];
                }
            }
        }

        return result;
    }
    
    @Override
    public Tensor to(Device device) {
        if (device == null) {
            return this;
        }

        return new GpuTensor(device, shape, data);
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
            throw new IllegalArgumentException("Inner dimensions must match: " + n + " != " + k);
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
