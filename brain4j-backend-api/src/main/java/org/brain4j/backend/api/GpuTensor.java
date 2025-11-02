package org.brain4j.backend.api;

import org.brain4j.backend.api.device.Device;
import org.brain4j.backend.api.memory.GpuMemory;

/**
 * Low-level representation of a tensor stored on a GPU backend.
 *
 * <p>Implementations provide access to the underlying device and raw
 * memory pointers used by native GPU kernels. This interface is intentionally
 * minimal and used by backend components that bind Java-side tensors to
 * device-side memory.
 *
 * @param <T> the native element type (usually Float or Double wrapper types)
 */
public interface GpuTensor<T> {

    /**
     * Number of elements contained in the tensor.
     *
     * @return element count
     */
    int size();

    /**
     * The device where this tensor resides.
     *
     * @return the {@link Device} instance
     */
    Device device();

    /**
     * Pointer to the device memory holding the tensor data.
     *
     * @return a {@link GpuMemory} instance for the tensor data
     */
    GpuMemory<T> dataPointer();

    /**
     * Pointer to the device memory holding the tensor shape.
     *
     * @return a {@link GpuMemory} instance for the shape array
     */
    GpuMemory<T> shapePointer();

    /**
     * Pointer to the device memory holding the tensor strides.
     *
     * @return a {@link GpuMemory} instance for the strides array
     */
    GpuMemory<T> stridesPointer();
}
