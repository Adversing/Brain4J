package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.CleanableTask;
import org.brain4j.math.gpu.TempObject;
import org.lwjgl.opencl.CL10;

import java.util.concurrent.atomic.AtomicLong;

public class TempBuffer extends TempObject<Long> {
    
    public TempBuffer(long value) {
        super(value);
        register(new CleanableTask(refCount) {
            @Override
            public void clean() {
                CL10.clReleaseMemObject(value);
            }
        });
    }
}
