package org.brain4j.math.gpu.memory;

import org.jocl.cl_mem;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.jocl.CL.clReleaseMemObject;

public class CollectableState implements Runnable {

    private final AtomicBoolean released = new AtomicBoolean(false);
    private final cl_mem[] buffers;

    public CollectableState(cl_mem... buffers) {
        this.buffers = buffers;
    }

    @Override
    public void run() {
        if (released.compareAndSet(false, true)) {
            for (cl_mem buffer : buffers) {
                clReleaseMemObject(buffer);
            }
        }
    }
}
