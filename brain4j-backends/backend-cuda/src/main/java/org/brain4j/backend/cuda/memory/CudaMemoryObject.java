package org.brain4j.backend.cuda.memory;

import jcuda.driver.CUdeviceptr;
import org.brain4j.backend.cuda.device.CudaDevice;
import org.brain4j.backend.device.Device;
import org.brain4j.backend.memory.MemoryObject;

import static jcuda.driver.JCudaDriver.*;

public class CudaMemoryObject implements MemoryObject<CUdeviceptr> {
    
    private final CudaDevice device;
    private final CUdeviceptr pointer;
    private final long size;

    public CudaMemoryObject(CudaDevice device, long size) {
        this.device = device;
        this.size = size;
        this.pointer = new CUdeviceptr();
        cuMemAlloc(pointer, size);
    }

    @Override
    public CUdeviceptr pointer() {
        return pointer;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public CudaDevice device() {
        return device;
    }
    
    @Override
    public void close() {
        cuMemFree(pointer);
    }
}