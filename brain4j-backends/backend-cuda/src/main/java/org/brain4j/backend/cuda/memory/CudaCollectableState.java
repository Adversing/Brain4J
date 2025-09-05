package org.brain4j.backend.cuda.memory;

import java.util.concurrent.atomic.AtomicBoolean;

public class CudaCollectableState implements Runnable {
    
    private final AtomicBoolean released = new AtomicBoolean(false);
    private final CudaMemoryObject[] memory;
    
    public CudaCollectableState(CudaMemoryObject... memory) {
        this.memory = memory;
    }
    
    @Override
    public void run() {
        if (released.compareAndSet(false, true)) {
            for (CudaMemoryObject mem : memory) {
                mem.close();
            }
        }
    }
}
