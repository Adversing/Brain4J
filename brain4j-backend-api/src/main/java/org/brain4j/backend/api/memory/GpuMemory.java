package org.brain4j.backend.api.memory;

/**
 * Represents a block of memory allocated on a GPU device.
 *
 * <p>This interface provides access to device memory allocations through
 * backend-specific pointer types. The memory is automatically released
 * when the object is closed, following RAII principles.
 *
 * <p>Implementations typically wrap native memory handles like OpenCL
 * memory objects or CUDA device pointers.
 *
 * @param <T> the backend-specific pointer type
 */
public interface GpuMemory<T> extends AutoCloseable {

    /**
     * Returns the backend-specific pointer to the device memory.
     *
     * @return native memory pointer/handle
     */
    T pointer();

    /**
     * Releases the device memory allocation.
     * <p>
     * This method should be called when the memory is no longer needed,
     * typically via try-with-resources.
     */
    @Override
    void close();
}