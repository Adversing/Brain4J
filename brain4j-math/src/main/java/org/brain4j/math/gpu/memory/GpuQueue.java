package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.GpuContext;
import org.jocl.cl_command_queue;

public record GpuQueue(cl_command_queue queue, boolean shouldClose) implements AutoCloseable {
    
    @Override
    public void close() {
        if (!shouldClose) return;
        GpuContext.closeQueue(queue);
    }
}
