package org.brain4j.backend.opencl.memory;

import org.brain4j.backend.device.Device;
import org.brain4j.backend.memory.MemoryObject;
import org.brain4j.backend.opencl.device.OpenCLDevice;
import org.jocl.Pointer;
import org.jocl.cl_mem;

import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clReleaseMemObject;

public class OpenCLMemoryObject implements MemoryObject<cl_mem> {
    
    private final OpenCLDevice device;
    private final cl_mem pointer;
    private final long size;
    
    public OpenCLMemoryObject(OpenCLDevice device, long size, int flags) {
        this(device, null, size, flags);
    }
    
    public OpenCLMemoryObject(OpenCLDevice device, Pointer pointer, long size, int flags) {
        this.device = device;
        this.size = size;
        
        cl_mem created = clCreateBuffer(
            device.getContext(), flags, size, pointer, null
        );
        
        if (created == null) {
            throw new RuntimeException("Failed to allocate OpenCL buffer");
        }
        
        this.pointer = created;
    }
    
    @Override
    public cl_mem pointer() {
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
    public void close() {
        clReleaseMemObject(pointer);
    }
}
