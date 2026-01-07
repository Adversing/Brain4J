package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.TempObject;
import org.lwjgl.opencl.CL10;

import java.util.concurrent.atomic.AtomicInteger;

public class TempBuffer extends TempObject<Long> {

    public static AtomicInteger totalCreations = new AtomicInteger(0);

    public TempBuffer(long value) {
        super(value, () -> {
            int refs = totalCreations.decrementAndGet();
            CL10.clReleaseMemObject(value);
            System.out.println("After releases: " + refs);
        });
        int refs = totalCreations.incrementAndGet();
        System.out.println("after creation: " + refs);
    }
}
