package org.brain4j.backend.api.operator;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;

public interface BackendOperator<T> {
    GpuTensor<T> createTensor(T value);
    void matmul(Device device, GpuTensor<T> a, GpuTensor<T> b, GpuTensor<T> c);
    void add(Device device, GpuTensor<T> a, GpuTensor<T> b);
    void sub(Device device, GpuTensor<T> a, GpuTensor<T> b);
    void mul(Device device, GpuTensor<T> a, GpuTensor<T> b);
}
