package org.brain4j.math.clipper.impl;

import org.brain4j.math.Commons;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.clipper.GradientClipper;
import org.jocl.cl_kernel;

public class HardClipper implements GradientClipper {

    private double bound;
    
    public HardClipper() {
    }
    
    public HardClipper(double bound) { this.bound = bound; }
    
    @Override
    public void clipCpu(CpuTensor grad) {
        grad.map(x -> Commons.clamp(x, -bound, bound));
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        Device device = grad.device();
        cl_kernel kernel = GpuContext.kernel(device, kernelName());
        
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(kernel)
                .addMemParam(grad.dataBuffer())
                .addFloatParam((float) bound)
                .addIntParam(grad.size())
                .launch(queue, 1, grad.size());
        }
    }

    @Override
    public String kernelName() {
        return "hard_clip";
    }
}