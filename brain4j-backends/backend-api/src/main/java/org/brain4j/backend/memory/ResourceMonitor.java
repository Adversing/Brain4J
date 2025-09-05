package org.brain4j.backend.memory;

import java.util.concurrent.atomic.AtomicBoolean;

public class ResourceMonitor implements Runnable {
    
    private final AtomicBoolean released = new AtomicBoolean(false);
    private final MemoryObject<?>[] memoryObjects;
    
    public ResourceMonitor(MemoryObject<?>... memoryObjects) {
        this.memoryObjects = memoryObjects;
    }
    
    @Override
    public void run() {
        if (released.compareAndSet(false, true)) {
            for (MemoryObject<?> mem : memoryObjects) {
                mem.close();
            }
        }
    }
}
