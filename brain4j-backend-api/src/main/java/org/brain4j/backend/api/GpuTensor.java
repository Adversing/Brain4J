package org.brain4j.backend.api;

import org.brain4j.backend.api.device.Device;
import org.brain4j.backend.api.memory.GpuMemory;

public interface GpuTensor<T> {
    int size();
    Device device();
    GpuMemory<T> dataPointer();
    GpuMemory<T> shapePointer();
    GpuMemory<T> stridesPointer();
}
