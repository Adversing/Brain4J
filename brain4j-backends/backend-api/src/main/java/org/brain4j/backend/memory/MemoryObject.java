package org.brain4j.backend.memory;

import org.brain4j.backend.device.Device;

public interface MemoryObject<T> extends AutoCloseable {
    T pointer();
    
    long size();
    
    Device device();
    
    @Override
    void close();
}
