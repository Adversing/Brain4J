package org.brain4j.backend.api.operator;

import org.brain4j.backend.api.GpuTensor;
import org.brain4j.backend.api.device.Device;

import java.util.List;

/**
 * Interface for backend-specific tensor operations.
 *
 * <p>This interface defines the contract for implementing GPU-accelerated
 * tensor operations on specific compute devices (OpenCL, CUDA, etc).
 * Backend implementations must provide efficient native implementations
 * of common tensor operations like matrix multiplication and elementwise
 * arithmetic.
 *
 * <p>The type parameter T specifies the concrete tensor implementation
 * used by the backend (e.g., CUDATensor, OpenCLTensor).
 *
 * @param <T> the concrete tensor type for this backend
 */
public interface BackendOperator<T extends GpuTensor<?>> {
    
    /**
     * Returns the first available compute device.
     *
     * @return the first available device
     * @throws IllegalStateException if no devices are found
     */
    default Device firstDevice() {
        List<Device> devices = retrieveDevices();
        
        if (devices.isEmpty()) {
            throw new IllegalStateException("No devices were found!");
        }
        
        return devices.getFirst();
    }
    
    /**
     * Returns the number of available compute devices.
     *
     * @return count of available devices
     */
    int countDevices();

    /**
     * Retrieves all available compute devices for this backend.
     *
     * @return list of available devices
     */
    List<Device> retrieveDevices();

    /**
     * Creates a new tensor on the specified device.
     *
     * @param device target compute device
     * @param shape tensor dimensions
     * @param data initial tensor data
     * @return new tensor instance
     */
    T createTensor(Device device, int[] shape, float... data);

    /**
     * Matrix multiplication: c = a * b
     *
     * @param device compute device to use
     * @param a first input tensor
     * @param b second input tensor
     * @param c output tensor
     */
    void matmul(Device device, T a, T b, T c);

    /**
     * Element-wise addition: b = a + b
     *
     * @param device compute device to use
     * @param a first input tensor
     * @param b second input/output tensor
     */
    void add(Device device, T a, T b);

    /**
     * Element-wise subtraction: b = a - b
     *
     * @param device compute device to use
     * @param a first input tensor
     * @param b second input/output tensor
     */
    void sub(Device device, T a, T b);

    /**
     * Element-wise multiplication: b = a * b
     *
     * @param device compute device to use
     * @param a first input tensor
     * @param b second input/output tensor
     */
    void mul(Device device, T a, T b);
}
