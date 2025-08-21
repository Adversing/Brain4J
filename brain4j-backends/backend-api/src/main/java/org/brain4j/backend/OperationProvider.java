package org.brain4j.backend;

import org.brain4j.math.activation.Activation;

public interface OperationProvider {
    void matmul(GpuTensor a, GpuTensor b, GpuTensor c);

    void add(GpuTensor a, GpuTensor b, GpuTensor c);

    void add(GpuTensor a, double value);

    void sub(GpuTensor a, GpuTensor b, GpuTensor c);

    void sub(GpuTensor a, double value);

    void mul(GpuTensor a, GpuTensor b, GpuTensor c);

    void mul(GpuTensor a, double value);

    void div(GpuTensor a, GpuTensor b, GpuTensor c);

    void div(GpuTensor a, double value);

    void activate(Activation activation, GpuTensor in, GpuTensor out);
}
