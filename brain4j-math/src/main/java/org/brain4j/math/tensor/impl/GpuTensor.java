package org.brain4j.math.tensor.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.gpu.memory.CollectableState;
import org.brain4j.math.tensor.Tensor;
import org.jocl.*;

import java.lang.ref.Cleaner;
import java.nio.IntBuffer;
import java.util.Arrays;

import static org.jocl.CL.*;

public class GpuTensor extends BaseTensor {

    /* Garbage collector stuff */
    private static final Cleaner CLEANER = Cleaner.create();
    
    private final Device device;
    private final cl_mem shapeBuffer;
    private final cl_mem stridesBuffer;
    private final cl_mem dataBuffer;
    private final int size;

    public GpuTensor(Device device, int[] shape, float... data) {
        this.device = device;
        this.size = data.length == 0 ? Tensors.computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        cl_context context = device.context();

        long shapeSize = (long) Sizeof.cl_int * shape.length;
        long stridesSize = (long) Sizeof.cl_int * strides.length;
        long dataSize  = (long) Sizeof.cl_float * this.size;

        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

        this.shapeBuffer = clCreateBuffer(context, readFlag, shapeSize, Pointer.to(shape), null);
        this.stridesBuffer = clCreateBuffer(context, readFlag, stridesSize, Pointer.to(strides), null);

        Pointer dataPointer = data.length > 0 ? Pointer.to(data) : null;
        long writeFlag = data.length > 0 ? CL_MEM_COPY_HOST_PTR : 1;

        this.dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | writeFlag, dataSize, dataPointer, null);
        CLEANER.register(this, new CollectableState(dataBuffer, shapeBuffer, stridesBuffer));
    }

    public GpuTensor(Device device, int[] shape, cl_mem otherBuffer) {
        this.device = device;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        cl_context context = device.context();

        long shapeSize = (long) Sizeof.cl_int * shape.length;
        long stridesSize = (long) Sizeof.cl_int * strides.length;
        long dataSize  = (long) Sizeof.cl_float * this.size;

        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

        this.shapeBuffer = clCreateBuffer(context, readFlag, shapeSize, Pointer.to(shape), null);
        this.stridesBuffer = clCreateBuffer(context, readFlag, stridesSize, Pointer.to(strides), null);
        this.dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, null, null);
        
        CLEANER.register(this, new CollectableState(dataBuffer, shapeBuffer, stridesBuffer));
        
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            clEnqueueCopyBuffer(queue.clQueue(), otherBuffer, this.dataBuffer, 0, 0, dataSize,
                0, null, null);
        }
    }

    public Device device() {
        return device;
    }

    public cl_mem dataBuffer() {
        return dataBuffer;
    }

    public cl_mem stridesBuffer() {
        return stridesBuffer;
    }

    public cl_mem shapeBuffer() {
        return shapeBuffer;
    }

    public int size() {
        return size;
    }

    public static void initKernels(Device device) {
        cl_context context = device.context();

        cl_program tensorOpsProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/tensor_ops.cl");
        cl_program elementaryOpsProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/elementary_ops.cl");

        String[] tensorOpsKernels = { "matmul_batched", "add", "sub", "mul", "div", "transpose", "sum_along_dim", "layer_norm", "softmax_last_dim" };

        for (String kernel : tensorOpsKernels) {
            GpuContext.register(device, kernel, tensorOpsProgram);
        }

        String[] scalarKernels = { "add_scalar", "mul_scalar", "div_scalar", "pow_scalar", "sqrt" };

        for (String kernel : scalarKernels) {
            GpuContext.register(device, kernel, elementaryOpsProgram);
        }
    }

    private long roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) return globalSize;
        return globalSize + groupSize - r;
    }

    private Tensor launchScalarKernel(String kernelName, float value) {
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, kernelName)
                .addMemParam(dataBuffer)
                .addFloatParam(value)
                .addIntParam(size)
                .launch(queue, 1, size);
        }

        return this;
    }

    private Tensor launchElementaryKernel(String kernelName, Tensor other) {
        if (!(other instanceof GpuTensor)) {
            other = other.gpu(device);
        }

        GpuTensor B = (GpuTensor) other;

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, kernelName)
                .addMemParam(dataBuffer)
                .addMemParam(B.dataBuffer)
                .addIntParam(size)
                .addIntParam(broadcastDim)
                .addIntParam(batch)
                .launch(queue, 1, size);
        }

        return this;
    }

    @Override
    public Tensor clone() {
        return new GpuTensor(device, shape, this.dataBuffer);
    }

    @Override
    public Tensor to(Device device) {
        if (device == null) {
            return new CpuTensor(shape, data());
        }

        return this;
    }

    @Override
    public Tensor transpose() {
        if (rank() == 1) {
            return reshape(1, elements());
        }

        if (shape.length != 2) {
            throw new UnsupportedOperationException(
                "transpose() is supported only for 2D tensors, not for tensors with " + shape.length + " dimensions"
            );
        }

        int rows = shape[0];
        int cols = shape[1];

        GpuTensor result = Tensors.matrix(cols, rows).gpu(device);

        if (usesGrad()) {
            result.setAutogradContext(autogradContext);
        }

        int inRowStride = strides[0];
        int inColStride = strides[1];
        int outRowStride = result.strides[0];
        int outColStride = result.strides[1];

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, "transpose")
                .addMemParam(dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(rows)
                .addIntParam(cols)
                .addIntParam(inRowStride)
                .addIntParam(inColStride)
                .addIntParam(outRowStride)
                .addIntParam(outColStride)
                .launch(queue, 2, rows, cols);
        }

        return result;
    }

    @Override
    public Tensor add(Tensor other) {
        return launchElementaryKernel("add", other);
    }

    @Override
    public Tensor add(double value) {
        return launchScalarKernel("add_scalar", (float) value);
    }

    @Override
    public Tensor sub(Tensor other) {
        return launchElementaryKernel("sub", other);
    }

    @Override
    public Tensor sub(double value) {
        return launchScalarKernel("sub_scalar", (float) value);
    }

    @Override
    public Tensor mul(Tensor other) {
        return launchElementaryKernel("mul", other);
    }

    @Override
    public Tensor mul(double value) {
        return launchScalarKernel("mul_scalar", (float) value);
    }

    @Override
    public Tensor div(Tensor other) {
        return launchElementaryKernel("div", other);
    }

    @Override
    public Tensor div(double value) {
        return launchScalarKernel("div_scalar", (float) value);
    }

    @Override
    public Tensor pow(double value) {
        return launchScalarKernel("pow_scalar", (float) value);
    }

    @Override
    public Tensor sqrt() {
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, "sqrt")
                .addMemParam(dataBuffer)
                .addIntParam(size)
                .launch(queue, 1, size);
        }

        return this;
    }

    @Override
    public Tensor activate(Activation activation) {
        return super.activate(activation);
    }
    @Override
    public Tensor matmul(Tensor other) {
        if (!(other instanceof GpuTensor B)) {
            throw new IllegalArgumentException("Other tensor is not an instance of TensorGPU.");
        }
        
        int[] shapeA = shape();
        int[] shapeB = other.shape();
        
        if (shapeA.length < 2 || shapeB.length < 2) {
            throw new IllegalArgumentException("Both tensors must have rank >= 2.");
        }
        
        int M = shapeA[shapeA.length - 2];
        int K = shapeA[shapeA.length - 1];
        int Kb = shapeB[shapeB.length - 2];
        int P = shapeB[shapeB.length - 1];
        
        if (K != Kb) {
            throw new IllegalArgumentException("Incompatible inner dims for matmul: K != Kb (" + K + " != " + Kb + ")");
        }
        
        int aBatchRank = shapeA.length - 2;
        int bBatchRank = shapeB.length - 2;
        int maxBatchRank = Math.max(aBatchRank, bBatchRank);
        
        int[] aBatch = new int[maxBatchRank];
        int[] bBatch = new int[maxBatchRank];
        for (int i = 0; i < maxBatchRank; ++i) {
            int ai = i - (maxBatchRank - aBatchRank);
            int bi = i - (maxBatchRank - bBatchRank);
            aBatch[i] = (ai >= 0) ? shapeA[ai] : 1;
            bBatch[i] = (bi >= 0) ? shapeB[bi] : 1;
        }
        
        int[] outBatch = new int[maxBatchRank];
        long batchCountLong = 1;
        for (int i = 0; i < maxBatchRank; ++i) {
            int da = aBatch[i];
            int db = bBatch[i];
            
            if (da == db || da == 1 || db == 1) {
                outBatch[i] = Math.max(da, db);
            } else {
                throw new IllegalArgumentException("Cannot broadcast batch dimension: " + Arrays.toString(aBatch) +
                    " vs " + Arrays.toString(bBatch));
            }
            
            batchCountLong *= outBatch[i];
            
            if (batchCountLong > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Batch size too large");
            }
        }
        int batchCount = (int) batchCountLong;
        
        int[] outShape = new int[maxBatchRank + 2];
        System.arraycopy(outBatch, 0, outShape, 0, maxBatchRank);
        
        outShape[maxBatchRank] = M;
        outShape[maxBatchRank + 1] = P;
        
        GpuTensor result = new GpuTensor(device, outShape);
        
        int matrixSizeA = M * K;
        int matrixSizeB = K * P;
        int matrixSizeC = M * P;
        
        int[] outStrides = new int[maxBatchRank];
        int a = 1;
        
        for (int i = maxBatchRank - 1; i >= 0; --i) {
            outStrides[i] = a;
            a *= outBatch[i];
        }
        
        int[] offsetsA = new int[batchCount];
        int[] offsetsB = new int[batchCount];
        int[] offsetsC = new int[batchCount];
        
        for (int b = 0; b < batchCount; ++b) {
            int[] idx = new int[maxBatchRank];
            for (int i = 0; i < maxBatchRank; ++i) {
                idx[i] = (b / outStrides[i]) % outBatch[i];
            }
            
            int linearA = 0;
            for (int i = 0; i < aBatchRank; ++i) {
                int alignedPos = i + (maxBatchRank - aBatchRank);
                int dimSizeA = shapeA[i];
                int chosen = (dimSizeA == 1) ? 0 : idx[alignedPos];
                linearA = linearA * dimSizeA + chosen;
            }
            
            int linearB = 0;
            for (int i = 0; i < bBatchRank; ++i) {
                int alignedPos = i + (maxBatchRank - bBatchRank);
                int dimSizeB = shapeB[i];
                int chosen = (dimSizeB == 1) ? 0 : idx[alignedPos];
                linearB = linearB * dimSizeB + chosen;
            }
            
            offsetsA[b] = linearA * matrixSizeA;
            offsetsB[b] = linearB * matrixSizeB;
            offsetsC[b] = b * matrixSizeC;
        }
        
        final int TILE_SIZE = 16;
        long[] globalWorkSize = new long[] {
            roundUp(TILE_SIZE, M),
            roundUp(TILE_SIZE, P),
            batchCount
        };
        long[] localWorkSize = new long[] { TILE_SIZE, TILE_SIZE, 1 };
        
        IntBuffer offsetsABuf = IntBuffer.wrap(offsetsA);
        IntBuffer offsetsBBuf = IntBuffer.wrap(offsetsB);
        IntBuffer offsetsCBuf = IntBuffer.wrap(offsetsC);
        
        cl_context context = device.context();
        
        Pointer pointerA = Pointer.to(offsetsABuf);
        Pointer pointerB = Pointer.to(offsetsBBuf);
        Pointer pointerC = Pointer.to(offsetsCBuf);
        
        cl_mem memoryA = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, offsetsA.length, pointerA, null);
        cl_mem memoryB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, offsetsB.length, pointerB, null);
        cl_mem memoryC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, offsetsC.length, pointerC, null);
        
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "matmul_batched")
                .addMemParam(dataBuffer)
                .addMemParam(B.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addMemParam(memoryA)
                .addMemParam(memoryB)
                .addMemParam(memoryC)
                .addIntParam(M)
                .addIntParam(K)
                .addIntParam(P)
                .addIntParam(batchCount)
                .launch(queue, 3, globalWorkSize, localWorkSize);
        }
        
        return result;
    }
    
    @Override
    public Tensor sum(int dim, boolean keepDim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }

        int[] newShape = Tensors.computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];

        int outerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= shape[i];

        int innerSize = 1;
        for (int i = dim + 1; i < shape.length; i++) innerSize *= shape[i];

        GpuTensor result = new GpuTensor(device, newShape);

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "sum_along_dim")
                .addMemParam(dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(reducedSize)
                .addIntParam(outerSize)
                .addIntParam(innerSize)
                .launch(queue, 2, outerSize, innerSize);
        }

        return result;
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        int batchSize = 1;
        int featuresSize = shape[0];

        if (shape.length == 2) {
            batchSize = shape[0];
            featuresSize = shape[1];
        }

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, "layer_norm")
                .addMemParam(dataBuffer)
                .addIntParam(batchSize)
                .addIntParam(featuresSize)
                .addFloatParam((float) epsilon)
                .launch(queue, 1, batchSize);
        }

        return this;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];

        cl_command_queue queue = device.newCommandQueue();
        clEnqueueReadBuffer(
                queue,
                dataBuffer,
                CL_TRUE,
                0,
                (long) size * Sizeof.cl_float,
                Pointer.to(buffer),
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return buffer;
    }

    @Override
    public Tensor set(float value, int... indices) {
        return null;
    }

    @Override
    public int elements() {
        return size;
    }

    @Override
    public Tensor softmax() {
        return super.softmax();
    }

    @Override
    public Tensor softmax(double temperature) {
        GpuTensor result = new GpuTensor(device, shape);

        int lastDim = shape[shape.length - 1];
        int rows = size / lastDim;

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory
                .create(device, "softmax_last_dim")
                .addMemParam(dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(lastDim)
                .addFloatParam((float) temperature)
                .launch(queue, 1, rows);
        }

        return result;
    }
}
