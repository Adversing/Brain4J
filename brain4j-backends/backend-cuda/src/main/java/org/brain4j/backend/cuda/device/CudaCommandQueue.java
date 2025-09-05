package org.brain4j.backend.cuda.device;

import jcuda.driver.CUstream;
import org.brain4j.backend.device.CommandQueue;

import static jcuda.driver.JCudaDriver.*;

public class CudaCommandQueue implements CommandQueue {
    
    private final CUstream stream;
    
    public CudaCommandQueue() {
        this.stream = new CUstream();
        cuStreamCreate(stream, 0);
    }
    
    @Override
    public void synchronize() {
        cuStreamSynchronize(stream);
    }
    
    @Override
    public void close() {
        cuStreamDestroy(stream);
    }
}
