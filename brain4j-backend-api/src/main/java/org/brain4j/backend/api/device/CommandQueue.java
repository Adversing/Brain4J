package org.brain4j.backend.api.device;

public interface CommandQueue extends AutoCloseable {
    void synchronize();
    @Override void close();
}