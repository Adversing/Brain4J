package org.brain4j.math.gpu.kernel;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.gpu.memory.TempBuffer;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL10;
import org.lwjgl.system.MemoryStack;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class KernelFactory {

    private final long kernel;
    private int arguments;

    protected KernelFactory(long kernel) {
        this.kernel = kernel;
    }

    public static KernelFactory create(long kernel) {
        return new KernelFactory(kernel);
    }

    public static KernelFactory create(Device device, String kernelName) {
        return create(GpuContext.kernel(device, kernelName));
    }

    public KernelFactory addIntParam(int variable) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer buf = stack.ints(variable);
            CL10.clSetKernelArg(kernel, arguments++, buf);
        }
        return this;
    }

    public KernelFactory addFloatParam(float variable) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer buf = stack.floats(variable);
            CL10.clSetKernelArg(kernel, arguments++, buf);
        }
        return this;
    }

    public KernelFactory addMemParam(long memory) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer buf = stack.pointers(memory);
            CL10.clSetKernelArg(kernel, arguments++, buf);
        }
        return this;
    }

    public KernelFactory addMemParam(TempBuffer memory) {
        return addMemParam(memory.value());
    }

    public void launch(GpuQueue queue, int workDim, long... globalWorkSize) {
        launch(queue.pointer(), workDim, globalWorkSize);
    }

    public void launch(GpuQueue queue, int workDim, long[] globalWorkSize, long... localWorkSize) {
        launch(queue.pointer(), workDim, globalWorkSize, localWorkSize);
    }

    public void launch(long queue, int workDim, long... globalWorkSize) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer globalWorkBuf = stack.mallocPointer(workDim);
            for (long g : globalWorkSize) globalWorkBuf.put(g);
            globalWorkBuf.flip();

            int err = CL10.clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkBuf,
                null, null, null);
            DeviceUtils.checkError("launch_kernel", err);
        }
    }

    public void launch(long queue, int workDim, long[] globalWorkSize, long... localWorkSize) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer globalWorkBuf = stack.mallocPointer(workDim);
            for (long g : globalWorkSize) globalWorkBuf.put(g);
            globalWorkBuf.flip();

            PointerBuffer localWorkBuf = stack.mallocPointer(workDim);
            for (long g : localWorkSize) localWorkBuf.put(g);
            localWorkBuf.flip();
            
            int err = CL10.clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkBuf, localWorkBuf,
                null, null);
            DeviceUtils.checkError("launch_kernel", err);
        }
    }
}
