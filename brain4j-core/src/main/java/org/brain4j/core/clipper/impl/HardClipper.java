package org.brain4j.core.clipper.impl;

import org.brain4j.common.device.Device;
import org.brain4j.common.kernel.GpuContextHandler;
import org.brain4j.common.kernel.KernelFactory;
import org.brain4j.common.tensor.impl.CpuTensor;
import org.brain4j.common.tensor.impl.GpuTensor;
import org.brain4j.core.clipper.GradientClipper;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public void clipCpu(CpuTensor grad) {
        grad.map(x -> Math.max(-bound, Math.min(bound, x)));
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        Device device = grad.device();

        cl_kernel kernel = GpuContextHandler.kernel(device, kernelName());
        cl_command_queue queue = GpuContextHandler.queue(device);

        KernelFactory
            .create(kernel)
            .addMemParam(grad.dataBuffer())
            .addFloatParam((float) bound)
            .addIntParam(grad.size())
            .launch(queue, 1, grad.size());
    }

    @Override
    public String kernelName() {
        return "hard_clip";
    }
}