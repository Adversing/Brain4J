package org.brain4j.math.gpu.device;

import org.brain4j.math.gpu.memory.GpuQueue;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

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
            ", pointer=" + queue +
            '}';
    }
    
    public void printLimits() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer lb = stack.mallocPointer(1);
            
            // max work group size
            CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_MAX_WORK_GROUP_SIZE, lb, null);
            System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE = " + lb.get(0));
            
            // max work item sizes (vector of size)
            PointerBuffer p = stack.mallocPointer(3);
            CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_MAX_WORK_ITEM_SIZES, p, null);
            System.out.println("CL_DEVICE_MAX_WORK_ITEM_SIZES = " + p.get(0) + " " + p.get(1) + " " + p.get(2));
            
            // local mem size
            CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_LOCAL_MEM_SIZE, lb, null);
            System.out.println("CL_DEVICE_LOCAL_MEM_SIZE = " + lb.get(0));
            
            CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_MAX_MEM_ALLOC_SIZE, lb, null);
            System.out.println("CL_DEVICE_MAX_MEM_ALLOC_SIZE = " + lb.get(0));
            
            CL10.clGetDeviceInfo(device, CL10.CL_DEVICE_GLOBAL_MEM_SIZE, lb, null);
            System.out.println("CL_DEVICE_GLOBAL_MEM_SIZE = " + lb.get(0));
        }
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
        int[] err = new int[1];
        long result = CL10.clCreateCommandQueue(context, device, 0, err);
        
        DeviceUtils.checkError("create_command_queue", err[0]);
        return result;
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
        int[] err = new int[1];
        long buffer = CL10.clCreateBuffer(context, flags, data, err);
        
        DeviceUtils.checkError("create_buffer", err[0]);
        return buffer;
    }

    public void createQueue() {
        this.queue = new GpuQueue(newCommandQueue(), false);
    }
}
