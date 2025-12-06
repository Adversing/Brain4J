package org.brain4j.math.gpu.device;

import org.brain4j.math.gpu.memory.GpuQueue;
import org.jocl.*;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;
import java.time.LocalDate;

import static org.jocl.CL.*;

public class Device {

    private final long platform;
    private final long device;
    private final long context;
    private GpuQueue queue;

    public Device(long platformAddr, long deviceAddr) {
        this.platform = platformAddr;
        this.device = deviceAddr;
        this.context = newContext();
    }

    @Override
    public String toString() {
        return "Device{" +
            "platform=" + platform +
            ", device=" + device +
            ", context=" + context +
            ", queue=" + queue +
            '}';
    }

    public long newContext() {
        try (MemoryStack stack = MemoryStack.stackPush()) {

            PointerBuffer properties = stack.mallocPointer(3);

            properties.put(CL10.CL_CONTEXT_PLATFORM).put(platform).put(0);
            properties.flip();
            return CL10.clCreateContext(properties, device, null, 0, null);
        }
    }

    public long newCommandQueue() {
        return CL10.clCreateCommandQueue(context, device, 0, (IntBuffer) null);
    }

    public String name() {
        return DeviceUtils.deviceName(device);
    }

    public long platform() {
        return platform;
    }

    public long device() {
        return device;
    }

    public long context() {
        return context;
    }

    public GpuQueue queue() {
        return queue;
    }

    public void setQueue(GpuQueue queue) {
        this.queue = queue;
    }

    public long createBuffer(long flags, float[] data) {
        int[] err = new int[1];

        long buffer = CL10.clCreateBuffer(context, flags, data, err);
        DeviceUtils.checkError("create_buffer", err[0]);

        return buffer;
    }

    public long createBuffer(long flags, long dataSize) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer err = stack.mallocInt(1);

            long buffer = CL10.clCreateBuffer(context, flags, dataSize, err);
            DeviceUtils.checkError("create_buffer", err.get(0));

            return buffer;
        }
    }

    public long createBuffer(long flags, int[] data) {
        return CL10.clCreateBuffer(context, flags, data, null);
    }

    public void createQueue() {
        this.queue = new GpuQueue(this, newCommandQueue(), false);
    }
}
