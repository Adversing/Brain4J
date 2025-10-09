package org.brain4j.backend.api;

import org.brain4j.backend.api.device.Device;

public interface GpuMemory<T> extends AutoCloseable {
    T pointer();
    @Override void close();
}