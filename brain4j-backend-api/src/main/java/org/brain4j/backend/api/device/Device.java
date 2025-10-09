package org.brain4j.backend.api.device;

public interface Device {
    String name() throws Throwable;
    CommandQueue newCommandQueue();
}