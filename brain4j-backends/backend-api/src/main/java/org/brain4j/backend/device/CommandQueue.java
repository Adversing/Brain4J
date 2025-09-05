package org.brain4j.backend.device;

public interface CommandQueue extends AutoCloseable {
    void synchronize();
    
    @Override
    void close();
}
