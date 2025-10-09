package org.brain4j.backend.api.operator;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;

import java.util.List;

public interface BackendOperator<T extends GpuTensor<?>> {
    
    default Device firstDevice() {
        List<Device> devices = retrieveDevices();
        
        if (devices.isEmpty()) {
            throw new IllegalStateException("No devices were found!");
        }
        
        return devices.getFirst();
    }
    
    int countDevices();
    List<Device> retrieveDevices();
    T createTensor(Device device, int[] shape, float... data);
    void matmul(Device device, T a, T b, T c);
    void add(Device device, T a, T b);
    void sub(Device device, T a, T b);
    void mul(Device device, T a, T b);
}
