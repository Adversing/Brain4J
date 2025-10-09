package org.brain4j.backend.cuda;

import org.brain4j.backend.api.device.CommandQueue;
import org.brain4j.backend.api.device.Device;
import org.cuda4j.device.CudaDevice;

public class CudaDeviceWrap implements Device {
    
    private final CudaDevice device;
    
    public CudaDeviceWrap(CudaDevice device) {
        this.device = device;
    }
    
    @Override
    public String name() throws Throwable {
        return device.getName();
    }
    
    @Override
    public CommandQueue newCommandQueue() {
        return null;
    }
}
