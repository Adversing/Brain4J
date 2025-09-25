package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.jocl.cl_command_queue;

public record GpuQueue(Device device, cl_command_queue queue, boolean shouldClose) implements AutoCloseable {
    
    @Override
    public void close() {
        if (!shouldClose) return;
        GpuContext.closeQueue(device);
    }
}
