package org.brain4j.backend.device;

public interface Device {
    String name();

    CommandQueue newCommandQueue();
}
