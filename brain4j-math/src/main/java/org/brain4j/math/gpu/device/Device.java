package org.brain4j.math.gpu.device;

import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.gpu.memory.TempBuffer;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;

import static org.lwjgl.opencl.CL10.*;

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
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, lb, null);
            System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE = " + lb.get(0));
            
            // max work item sizes (vector of size)
            PointerBuffer p = stack.mallocPointer(3);
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, p, null);
            System.out.println("CL_DEVICE_MAX_WORK_ITEM_SIZES = " + p.get(0) + " " + p.get(1) + " " + p.get(2));
            
            // local mem size
            clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, lb, null);
            System.out.println("CL_DEVICE_LOCAL_MEM_SIZE = " + lb.get(0));
            
            clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, lb, null);
            System.out.println("CL_DEVICE_MAX_MEM_ALLOC_SIZE = " + lb.get(0));
            
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, lb, null);
            System.out.println("CL_DEVICE_GLOBAL_MEM_SIZE = " + lb.get(0));
        }
    }

    public long newContext() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer properties = stack.mallocPointer(3);

            properties.put(CL_CONTEXT_PLATFORM).put(platform).put(0);
            properties.flip();
            
            return clCreateContext(properties, device, null, 0, null);
        }
    }

    public long newCommandQueue() {
        int[] err = new int[1];
        long result = clCreateCommandQueue(context, device, 0, err);
        
        DeviceUtils.checkError("create_command_queue", err[0]);
        return result;
    }

    public String name() {
        return DeviceUtils.deviceName(device);
    }

    public TempBuffer createBuffer(long flags, float[] data) {
        int[] err = new int[1];
        long buffer = clCreateBuffer(context, flags, data, err);
        
        DeviceUtils.checkError("create_buffer", err[0]);
        return new TempBuffer(buffer);
    }

    public TempBuffer createBuffer(long flags, long dataSize) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer err = stack.mallocInt(1);
            long buffer = clCreateBuffer(context, flags, dataSize, err);
            
            DeviceUtils.checkError("create_buffer", err.get(0));
            return new TempBuffer(buffer);
        }
    }

    public TempBuffer createBuffer(long flags, int[] data) {
        int[] err = new int[1];
        long buffer = clCreateBuffer(context, flags, data, err);
        
        DeviceUtils.checkError("create_buffer", err[0]);
        return new TempBuffer(buffer);
    }

    public void createQueue() {
        this.queue = new GpuQueue(newCommandQueue(), false);
    }
    
    public long getPlatform() {
        return platform;
    }
    
    public long getDevice() {
        return device;
    }
    
    public long getContext() {
        return context;
    }
    
    public GpuQueue getQueue() {
        return queue;
    }
    
    public void setQueue(GpuQueue queue) {
        this.queue = queue;
    }
}
