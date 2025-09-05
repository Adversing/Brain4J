package org.brain4j.backend.cuda;

import jcuda.Pointer;
import org.brain4j.backend.GpuTensor;
import org.brain4j.backend.cuda.device.CudaDevice;
import org.brain4j.backend.cuda.memory.CudaMemoryObject;
import org.brain4j.backend.memory.ResourceMonitor;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public class CudaTensor extends GpuTensor {
    
    private final CudaMemoryObject shapeBuffer;
    private final CudaMemoryObject stridesBuffer;
    private final CudaMemoryObject dataBuffer;
    
    public CudaTensor(CudaDevice device, int[] shape, float... data) {
        super(device);
        this.size = data.length == 0 ? Tensors.computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);
        
        long shapeSize = (long) INT_SIZE * shape.length;
        long stridesSize = (long) INT_SIZE * strides.length;
        long dataSize = (long) FLOAT_SIZE * this.size;
        
        this.shapeBuffer = new CudaMemoryObject(device, shapeSize);
        this.stridesBuffer = new CudaMemoryObject(device, stridesSize);
        this.dataBuffer = new CudaMemoryObject(device, dataSize);
        
        cuMemcpyHtoD(shapeBuffer.pointer(), Pointer.to(shape), shapeSize);
        cuMemcpyHtoD(stridesBuffer.pointer(), Pointer.to(strides), stridesSize);
        
        if (data.length > 0) {
            cuMemcpyHtoD(dataBuffer.pointer(), Pointer.to(data), dataSize);
        }
        
        CLEANER.register(this, new ResourceMonitor(dataBuffer, shapeBuffer, stridesBuffer));
    }
    
    @Override
    public CudaMemoryObject dataBuffer() {
        return dataBuffer;
    }
    
    @Override
    public CudaMemoryObject shapeBuffer() {
        return shapeBuffer;
    }
    
    @Override
    public CudaMemoryObject stridesBuffer() {
        return stridesBuffer;
    }
    
    @Override
    public Tensor to(Device device) {
        return null;
    }
    
    @Override
    public Tensor matmul(Tensor other) {
        return null;
    }
}
