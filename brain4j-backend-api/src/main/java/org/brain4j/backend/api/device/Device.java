package org.brain4j.backend.api.device;

/**
 * Represents a compute device capable of executing tensor operations.
 *
 * <p>A Device typically corresponds to a physical GPU or other compute
 * accelerator. It provides access to device-specific properties and
 * the ability to create command queues for executing operations.
 *
 * <p>Device implementations are backend-specific and may wrap native
 * resources like OpenCL devices or CUDA devices.
 */
public interface Device {

    /**
     * Returns the name/identifier of this device.
     *
     * @return device name as reported by the backend
     */
    String name();

    /**
     * Creates a new command queue for submitting work to this device.
     *
     * <p>Command queues are used to schedule tensor operations for
     * execution on the device. Multiple queues may exist for a single
     * device to support concurrent execution.
     *
     * @return a new command queue instance
     */
    CommandQueue newCommandQueue();
}