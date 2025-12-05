package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.TempObject;
import org.jocl.cl_mem;

import static org.jocl.CL.clReleaseMemObject;

public class TempBuffer extends TempObject<cl_mem> {
    
    public TempBuffer(cl_mem value) {
        super(value, () -> clReleaseMemObject(value));
    }
}
