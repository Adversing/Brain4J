package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.CleanableTask;
import org.brain4j.math.gpu.TempObject;
import org.lwjgl.opencl.CL10;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class TempBuffer extends TempObject<Long> {
    
    public TempBuffer(long value) {
        super(value);
        register(new CleanableTask(refCount) {
            @Override
            public void clean() {
                System.out.println("released " + value);
                CL10.clReleaseMemObject(value);
            }
        });
    }
}
