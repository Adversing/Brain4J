package org.brain4j.math.gpu;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class GpuContext {

    private static final Map<Device, ThreadLocal<GpuQueue>> queues = new HashMap<>();
    private static final Map<Device, Map<String, cl_kernel>> kernelCache = new HashMap<>();

    public static void register(Device device, String kernelName, cl_program program) {
        kernelCache
            .computeIfAbsent(device, d -> new HashMap<>())
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
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device);
        }

        return kernel;
    }

    public static void updateQueue(Device device, cl_command_queue newQueue) {
        queues
            .computeIfAbsent(device, d -> new ThreadLocal<>())
            .set(new GpuQueue(newQueue, false));
    }

    public static GpuQueue getOrCreate(Device device) {
        GpuQueue current = queue(device);

        if (current == null) {
            current = new GpuQueue(device.newCommandQueue(), true);
        }

        return current;
    }

    public static GpuQueue queue(Device device) {
        ThreadLocal<GpuQueue> localQueue = queues.get(device);

        if (localQueue == null || localQueue.get() == null) {
            throw new IllegalStateException("Command queue not set for device: " + device);
        }

        return localQueue.get();
    }

    public static void closeQueue(cl_command_queue queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }

    public static void closeQueue(Device device) {
        GpuQueue commandQueue = queue(device);
        closeQueue(commandQueue.clQueue());
    }
}