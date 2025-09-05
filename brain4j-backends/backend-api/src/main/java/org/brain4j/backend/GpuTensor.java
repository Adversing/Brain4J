package org.brain4j.backend;

import org.brain4j.backend.device.Device;
import org.brain4j.backend.memory.MemoryObject;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.BaseTensor;

import java.lang.ref.Cleaner;

public abstract class GpuTensor extends BaseTensor {
    
    public static final Cleaner CLEANER = Cleaner.create();
    public static int SHORT_SIZE = 2;
    public static int INT_SIZE = 4;
    public static int FLOAT_SIZE = 4;
    public static int LONG_SIZE = 8;
    public static int DOUBLE_SIZE = 8;
    
    protected Device device;
    protected int size;
    
    public GpuTensor(Device device) {
        this.device = device;
    }
    
    public Device device() {
        return device;
    }
    
    public int size() {
        return size;
    }
    
    public abstract MemoryObject dataBuffer();
    
    public abstract MemoryObject shapeBuffer();
    
    public abstract MemoryObject stridesBuffer();
}
