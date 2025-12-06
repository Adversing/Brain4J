package org.brain4j.math.gpu;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;
import org.jocl.cl_program;
import org.lwjgl.opencl.CL10;

import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class GpuContext {

    private static final Map<Device, Map<String, Long>> kernelCache = new HashMap<>();

    public static void register(Device device, String kernelName, long program) {
        kernelCache.computeIfAbsent(device, d -> new HashMap<>())
            .compute(kernelName, (name, existingKernel) -> {
                if (existingKernel != null) {
                    throw new IllegalArgumentException("Kernel " + name + " already initialized for device " + device);
                }

                return CL10.clCreateKernel(program, name, (IntBuffer) null);
            });
    }

    public static long kernel(Device device, String kernelName) {
        Map<String, Long> deviceKernels = kernelCache.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        long kernel = deviceKernels.getOrDefault(kernelName, -1L);

        if (kernel == -1) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.name());
        }

        return kernel;
    }

    public static GpuQueue getOrCreate(Device device) {
        GpuQueue queue = device.queue();

        if (queue == null) {
            long clQueue = device.newCommandQueue();
            queue = new GpuQueue(device, clQueue, true);
        }

        return queue;
    }

    public static void finishAndRelease(GpuQueue queue) {
        long clCommandQueue = queue.queue();

        if (clCommandQueue == 0) return;

        finishAndReleaseCl(clCommandQueue);
    }

    public static void finishAndReleaseCl(long commandQueue) {
        CL10.clFinish(commandQueue);
        CL10.clReleaseCommandQueue(commandQueue);
    }

    public static void finishAndRelease(Device device) {
        finishAndRelease(device.queue());
        device.setQueue(null);
    }
}