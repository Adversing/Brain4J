package org.brain4j.backend;

import org.brain4j.backend.device.Device;
import org.brain4j.backend.memory.MemoryObject;
import org.brain4j.math.tensor.Tensor;

public interface GpuTensor extends Tensor {
    int SHORT_SIZE = 2;
    int INT_SIZE = 4;
    int FLOAT_SIZE = 4;
    int LONG_SIZE = 8;
    int DOUBLE_SIZE = 8;

    Device device();

    MemoryObject dataBuffer();

    MemoryObject shapeBuffer();

    MemoryObject stridesBuffer();
}
