package org.brain4j.backend.opencl.device;

import org.brain4j.backend.device.CommandQueue;
import org.jocl.cl_command_queue;

import static org.jocl.CL.clFinish;
import static org.jocl.CL.clReleaseCommandQueue;

public class OpenCLCommandQueue implements CommandQueue {
    
    private final cl_command_queue queue;
    
    public OpenCLCommandQueue(cl_command_queue queue) {
        this.queue = queue;
    }
    
    @Override
    public void synchronize() {
        clFinish(queue);
    }
    
    @Override
    public void close() {
        clReleaseCommandQueue(queue);
    }
}
