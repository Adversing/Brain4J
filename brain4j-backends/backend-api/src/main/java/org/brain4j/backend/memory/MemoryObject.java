package org.brain4j.backend.memory;

import org.brain4j.backend.device.Device;

public interface MemoryObject extends AutoCloseable {
    long size();
    
    Device device();
    
    void copyFromHost(byte[] data);
    
    void copyToHost(byte[] dest);
    
    @Override
    void close();
}
