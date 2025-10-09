package org.brain4j.backend.cuda;

import org.brain4j.backend.api.GpuMemory;
import org.cuda4j.buffer.CudaBuffer;
import org.cuda4j.buffer.CudaPointer;

public class CudaMemory implements GpuMemory<CudaPointer> {
    
    private final CudaBuffer buffer;
    private final CudaPointer pointer;
    
    public CudaMemory(CudaBuffer buffer) throws Throwable {
        this.buffer = buffer;
        this.pointer = CudaPointer.fromBuffer(buffer);
    }
    
    @Override
    public CudaPointer pointer() {
        return pointer;
    }
    
    @Override
    public void close() {
        try {
            buffer.release();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}
