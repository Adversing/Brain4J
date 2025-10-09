package org.brain4j.backend.cuda;

import org.brain4j.backend.api.device.CommandQueue;
import org.brain4j.backend.api.device.Device;
import org.cuda4j.Cuda4J;
import org.cuda4j.context.CudaContext;
import org.cuda4j.device.CudaDevice;

import java.lang.ref.Cleaner;

public record CudaDeviceWrap(CudaDevice device, CudaContext context) implements Device {
    
    private final static Cleaner REGISTER = Cleaner.create();
    
    public CudaDeviceWrap(CudaDevice device, CudaContext context) {
        this.device = device;
        this.context = context;
        
        REGISTER.register(this, () -> Cuda4J.unchecked(context::release));
    }
    
    @Override
    public String name() {
        try {
            return device.getName();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public CommandQueue newCommandQueue() {
        return null;
    }
}
