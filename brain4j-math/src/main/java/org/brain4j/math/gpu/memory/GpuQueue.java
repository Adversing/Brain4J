package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.GpuContext;

public record GpuQueue(long pointer, boolean temporary) implements AutoCloseable {
    
    @Override
    public void close() {
        if (!temporary) return;
        
        GpuContext.finishAndRelease(pointer);
    }
}
