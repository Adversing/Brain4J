package org.brain4j.backend.cuda.device;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import org.brain4j.backend.device.CommandQueue;
import org.brain4j.backend.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;

import java.nio.charset.StandardCharsets;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;

public class CudaDevice implements Device {
    
    private final CUdevice device;
    
    public CudaDevice(CUdevice device) {
        this.device = device;
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }
    
    @Override
    public String name() {
        byte[] nameBytes = new byte[256];
        cuDeviceGetName(nameBytes, nameBytes.length, device);
        return new String(nameBytes, StandardCharsets.UTF_8).trim();
    }
    
    @Override
    public CommandQueue newCommandQueue() {
        return new CudaCommandQueue();
    }
}
