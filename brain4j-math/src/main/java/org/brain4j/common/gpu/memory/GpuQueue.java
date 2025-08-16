package org.brain4j.common.gpu.memory;

import org.brain4j.common.gpu.GpuContext;
import org.jocl.cl_command_queue;

public class GpuQueue implements AutoCloseable {

    private final cl_command_queue clQueue;
    private final boolean shouldClose;

    public GpuQueue(cl_command_queue clQueue, boolean shouldClose) {
        this.clQueue = clQueue;
        this.shouldClose = shouldClose;
    }

    @Override
    public void close() {
        if (!shouldClose) return;
        GpuContext.closeQueue(clQueue);
    }

    public cl_command_queue clQueue() {
        return clQueue;
    }
}
