package org.brain4j.backend.cuda.memory;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import org.brain4j.backend.device.Device;
import org.brain4j.backend.memory.MemoryObject;

import static jcuda.driver.JCudaDriver.*;

public class CudaMemoryObject implements MemoryObject {
    
    private final Device device;
    private final CUdeviceptr pointer;
    private final long size;

    public CudaMemoryObject(Device device, long size) {
        this.device = device;
        this.size = size;
        this.pointer = new CUdeviceptr();
        cuMemAlloc(pointer, size);
    }

    public CUdeviceptr pointer() {
        return pointer;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public void copyFromHost(byte[] data) {
        if (data.length > size) {
            throw new IllegalArgumentException("Data too large for buffer");
        }
        cuMemcpyHtoD(pointer, Pointer.to(data), data.length);
    }

    @Override
    public void copyToHost(byte[] dest) {
        if (dest.length > size) {
            throw new IllegalArgumentException("Destination too small");
        }
        cuMemcpyDtoH(Pointer.to(dest), pointer, dest.length);
    }
    
    @Override
    public void close() {
        cuMemFree(pointer);
    }
}