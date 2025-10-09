package org.brain4j.backend.cuda;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;
import org.cuda4j.buffer.CudaBuffer;
import org.cuda4j.buffer.CudaPointer;
import org.cuda4j.device.CudaDevice;

import java.util.stream.IntStream;

import static org.cuda4j.CudaObject.FLOAT_SIZE;
import static org.cuda4j.CudaObject.INT_SIZE;

public class CudaTensor implements GpuTensor<CudaPointer> {
    
    private final Device device;
    private final CudaMemory dataPointer;
    private final CudaMemory shapePointer;
    private final CudaMemory stridesPointer;
    private final int size;
    
    public CudaTensor(Device device, int[] shape, float... data) throws Throwable {
        this.device = device;
        this.size = computeSize(shape);
        
        int[] strides = computeStrides(shape);
        
        this.dataPointer = new CudaMemory(CudaBuffer.allocate(data, size * FLOAT_SIZE));
        this.shapePointer = new CudaMemory(CudaBuffer.allocate(shape, shape.length * INT_SIZE));
        this.stridesPointer = new CudaMemory(CudaBuffer.allocate(strides, strides.length * INT_SIZE));
    }
    
    public static int computeSize(int[] shape) {
        return IntStream.of(shape).reduce(1, (left, right) -> left * right);
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
    
    @Override
    public int size() {
        return size;
    }
    
    @Override
    public Device device() {
        return device;
    }
    
    @Override
    public CudaMemory dataPointer() {
        return dataPointer;
    }
    
    @Override
    public CudaMemory shapePointer() {
        return shapePointer;
    }
    
    @Override
    public CudaMemory stridesPointer() {
        return stridesPointer;
    }
}
