package org.brain4j.math.gpu.kernel;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.jocl.*;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.jocl.CL.*;
import static org.jocl.CL.CL_MEM_READ_WRITE;

public class KernelFactory {

    public record Argument(int index, int size, Pointer pointer) { }

    private final cl_kernel kernel;
    private final List<Argument> arguments;

    protected KernelFactory(cl_kernel kernel) {
        this.kernel = kernel;
        this.arguments = new ArrayList<>();
    }

    public static KernelFactory create(cl_kernel kernel) {
        return new KernelFactory(kernel);
    }

    public static KernelFactory create(Device device, String kernelName) {
        return create(GpuContext.kernel(device, kernelName));
    }

    public KernelFactory addIntParam(int variable) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_int, Pointer.to(new int[]{variable})));
        return this;
    }

    public KernelFactory addFloatParam(float variable) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_float, Pointer.to(new float[]{variable})));
        return this;
    }

    public KernelFactory addMemParam(cl_mem memory) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_mem, Pointer.to(memory)));
        return this;
    }

    public void launch(GpuQueue queue, int workDim, long... globalWorkSize) {
        launch(queue.clQueue(), workDim, globalWorkSize);
    }

    public void launch(GpuQueue queue, int workDim, long[] globalWorkSize, long... localWorkSize) {
        launch(queue.clQueue(), workDim, globalWorkSize, localWorkSize);
    }

    public void launch(cl_command_queue queue, int workDim, long... globalWorkSize) {
        for (Argument argument : arguments) {
            clSetKernelArg(kernel, argument.index, argument.size, argument.pointer);
        }

        clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkSize, null,
            0, null, null);
    }

    public void launch(cl_command_queue queue, int workDim, long[] globalWorkSize, long... localWorkSize) {
        for (Argument argument : arguments) {
            clSetKernelArg(kernel, argument.index, argument.size, argument.pointer);
        }

        clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkSize, localWorkSize,
            0, null, null);
    }
}
