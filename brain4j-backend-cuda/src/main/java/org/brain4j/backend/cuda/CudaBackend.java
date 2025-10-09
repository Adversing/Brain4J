package org.brain4j.backend.cuda;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;
import org.brain4j.backend.api.operator.BackendOperator;
import org.cuda4j.Cuda4J;
import org.cuda4j.buffer.CudaPointer;
import org.cuda4j.context.CudaContext;
import org.cuda4j.device.CudaDevice;

import java.util.ArrayList;
import java.util.List;

public class CudaBackend implements BackendOperator<CudaTensor> {

    public CudaBackend() {
        try {
            Cuda4J.init();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public int countDevices() {
        try {
            return Cuda4J.getDeviceCount();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public List<Device> retrieveDevices() {
        int count = countDevices();
        List<Device> devices = new ArrayList<>(count);
        
        for (int i = 0; i < count; i++) {
            try {
                CudaDevice device = CudaDevice.createSystemDevice(i);
                CudaContext context = device.createContext();
                
                CudaDeviceWrap wrap = new CudaDeviceWrap(device, context);
                devices.add(wrap);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
        
        return devices;
    }
    
    @Override
    public CudaTensor createTensor(Device device, int[] shape, float... data) {
        try {
            return new CudaTensor(device, shape, data);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override
    public void matmul(Device device, CudaTensor a, CudaTensor b, CudaTensor c) {
    
    }
    
    @Override
    public void add(Device device, CudaTensor a, CudaTensor b) {
    
    }
    
    @Override
    public void sub(Device device, CudaTensor a, CudaTensor b) {
    
    }
    
    @Override
    public void mul(Device device, CudaTensor a, CudaTensor b) {
    
    }
}
