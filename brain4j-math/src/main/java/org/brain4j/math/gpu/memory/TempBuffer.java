package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.TempObject;
import org.lwjgl.opencl.CL10;

public class TempBuffer extends TempObject<Long> {
    
    public TempBuffer(long value) {
        super(value, () -> CL10.clReleaseMemObject(value));
    }
}
