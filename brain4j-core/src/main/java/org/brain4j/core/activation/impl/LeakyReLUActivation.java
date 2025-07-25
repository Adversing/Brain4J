package org.brain4j.core.activation.impl;

import org.brain4j.common.activation.Activation;
import org.brain4j.common.gpu.kernel.KernelFactory;
import org.brain4j.common.tensor.impl.GpuTensor;
import org.brain4j.common.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.NormalHeInit;
import org.jocl.cl_kernel;

public class LeakyReLUActivation implements Activation {

    private final double alpha;

    public LeakyReLUActivation() {
        this(0.01);
    }

    public LeakyReLUActivation(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return Math.max(alpha * input, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : alpha;
    }

    @Override
    public String kernelPrefix() {
        return "leakyrelu";
    }

    @Override
    public KernelFactory createKernel(cl_kernel kernel, GpuTensor current, GpuTensor other) {
        return KernelFactory.create(kernel)
            .addMemParam(current.dataBuffer())
            .addMemParam(other.dataBuffer())
            .addFloatParam((float) alpha)
            .addIntParam(current.size());
    }
}
