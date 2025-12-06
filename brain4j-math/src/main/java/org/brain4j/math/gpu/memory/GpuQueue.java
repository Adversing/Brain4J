package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.GpuContext;

public record GpuQueue(long pointer, boolean shouldClose) implements AutoCloseable {
    
    @Override
    public void close() {
        if (!shouldClose) return;
        
        GpuContext.finishAndRelease(pointer);
    }
}
