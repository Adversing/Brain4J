package org.brain4j.common.activation;

import org.brain4j.common.gpu.memory.CloseableQueue;
import org.brain4j.common.gpu.device.Device;
import org.brain4j.common.gpu.GpuContext;
import org.brain4j.common.gpu.kernel.KernelFactory;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.Tensors;
import org.brain4j.common.tensor.impl.GpuTensor;
import org.brain4j.common.weightsinit.WeightInitialization;
import org.jocl.cl_kernel;

public interface Activation {

    /**
     * The default weight initialization for this activation function.
     * @return The default weight initialization.
     */
    WeightInitialization defaultWeightInit();

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     * @return the activation of the input
     */
    double activate(double input);

    /**
     * Gets the derivative (scalar) of the activation at a single value.
     * @return the derivative of the input as a double value
     */
    double derivative(double input);

    /**
     * Gets the kernel prefix for this activation
     * @return the prefix of the kernel
     */
    String kernelPrefix();

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String name() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }

    /**
     * Creates the kernel to execute.
     * @param kernel the OpenCL kernel instance
     * @param current the current tensor
     * @param other the resulting tensor
     * @return a kernel factory ready to be launched
     */
    default KernelFactory createKernel(cl_kernel kernel, GpuTensor current, GpuTensor other) {
        return KernelFactory
            .create(kernel)
            .addMemParam(current.dataBuffer())
            .addMemParam(other.dataBuffer())
            .addIntParam(current.size());
    }

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    default Tensor activate(Tensor input) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            Device device = gpuInput.device();
            GpuTensor result = new GpuTensor(device, gpuInput.shape());

            try (CloseableQueue queue = GpuContext.getOrCreate(device)) {
                cl_kernel kernel = GpuContext.kernel(device, kernelPrefix() + "_forward");

                KernelFactory factory = createKernel(kernel, gpuInput, result);
                factory.launch(queue, 1, gpuInput.size());
            }

            return result;
        }

        float[] resultData = new float[input.elements()];
        float[] inputData = input.data();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) activate(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Tensor derivative(Tensor input) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            Device device = gpuInput.device();
            GpuTensor result = new GpuTensor(device, gpuInput.shape());

            try (CloseableQueue queue = GpuContext.getOrCreate(device)) {
                cl_kernel kernel = GpuContext.kernel(device, kernelPrefix() + "_forward");

                KernelFactory factory = createKernel(kernel, gpuInput, result);
                factory.launch(queue, 1, gpuInput.size());
            }

            return result;
        }

        float[] resultData = new float[input.elements()];
        float[] inputData = input.data();

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) derivative(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }
}
