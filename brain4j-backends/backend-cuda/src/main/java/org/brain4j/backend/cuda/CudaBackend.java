package org.brain4j.backend.cuda;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;
import org.brain4j.backend.api.operator.BackendOperator;
import org.cuda4j.buffer.CudaPointer;

public class CudaBackend implements BackendOperator<CudaPointer> {
    @Override
    public GpuTensor<CudaPointer> createTensor(CudaPointer value) {
        
        return null;
    }
    
    @Override
    public void matmul(Device device, GpuTensor<CudaPointer> a, GpuTensor<CudaPointer> b, GpuTensor<CudaPointer> c) {
    
    }
    
    @Override
    public void add(Device device, GpuTensor<CudaPointer> a, GpuTensor<CudaPointer> b) {
    
    }
    
    @Override
    public void sub(Device device, GpuTensor<CudaPointer> a, GpuTensor<CudaPointer> b) {
    
    }
    
    @Override
    public void mul(Device device, GpuTensor<CudaPointer> a, GpuTensor<CudaPointer> b) {
    
    }
}
