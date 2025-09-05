package org.brain4j.backend.opencl;

import org.brain4j.backend.GpuTensor;
import org.brain4j.backend.memory.ResourceMonitor;
import org.brain4j.backend.opencl.device.OpenCLDevice;
import org.brain4j.backend.opencl.memory.OpenCLMemoryObject;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.jocl.Pointer;

import static org.jocl.CL.*;

public class OpenCLTensor extends GpuTensor {
    
    private final OpenCLMemoryObject shapeBuffer;
    private final OpenCLMemoryObject stridesBuffer;
    private final OpenCLMemoryObject dataBuffer;
    
    public OpenCLTensor(OpenCLDevice device, int[] shape, float... data) {
        super(device);
        this.size = data.length == 0 ? Tensors.computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);
        
        long shapeSize = (long) INT_SIZE * shape.length;
        long stridesSize = (long) INT_SIZE * strides.length;
        long dataSize = (long) FLOAT_SIZE * this.size;
        
        int flags = (int) (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        int writeFlags = (int) (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
        Pointer dataPointer = data.length > 0 ? Pointer.to(data) : null;
        
        this.shapeBuffer = new OpenCLMemoryObject(device, Pointer.to(shape), shapeSize, flags);
        this.stridesBuffer = new OpenCLMemoryObject(device, Pointer.to(strides), stridesSize, flags);
        this.dataBuffer = new OpenCLMemoryObject(device, dataPointer, dataSize, writeFlags);
        
        CLEANER.register(this, new ResourceMonitor(dataBuffer, shapeBuffer, stridesBuffer));
    }
    
    @Override
    public OpenCLMemoryObject dataBuffer() {
        return dataBuffer;
    }
    
    @Override
    public OpenCLMemoryObject shapeBuffer() {
        return shapeBuffer;
    }
    
    @Override
    public OpenCLMemoryObject stridesBuffer() {
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
