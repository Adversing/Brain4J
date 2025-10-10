package org.brain4j.backend.api;

public interface GpuMemory<T> extends AutoCloseable {
    T pointer();
    @Override void close();
}