package org.brain4j.math.gpu;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.jocl.cl_kernel;
import org.jocl.cl_command_queue;
import org.jocl.cl_program;

import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class GpuContext {

    private static final Map<Device, Map<String, cl_kernel>> kernelCache = new HashMap<>();

    public static void register(Device device, String kernelName, cl_program program) {
        kernelCache.computeIfAbsent(device, d -> new HashMap<>())
            .compute(kernelName, (name, existingKernel) -> {
                if (existingKernel != null) {
                    throw new IllegalArgumentException("Kernel " + name + " already initialized for device " + device);
                }

                return clCreateKernel(program, name, null);
            });
    }

    public static cl_kernel kernel(Device device, String kernelName) {
        Map<String, cl_kernel> deviceKernels = kernelCache.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        cl_kernel kernel = deviceKernels.get(kernelName);

        if (kernel == null) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.name());
        }

        return kernel;
    }

    public static GpuQueue getOrCreate(Device device) {
        GpuQueue queue = device.queue();

        if (queue == null) {
            cl_command_queue clQueue = device.newCommandQueue();
            queue = new GpuQueue(device, clQueue,true);
        }

        return queue;
    }

    public static void finishAndRelease(GpuQueue queue) {
        cl_command_queue clCommandQueue = queue.queue();

        if (clCommandQueue == null) return;

        finishAndReleaseCl(clCommandQueue);
    }

    public static void finishAndReleaseCl(cl_command_queue queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }

    public static void finishAndRelease(Device device) {
        finishAndRelease(device.queue());
        device.setQueue(null);
    }
}