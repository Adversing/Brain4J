package org.brain4j.math.gpu;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public abstract class CleanableTask implements Runnable {
    
    private final AtomicBoolean released = new AtomicBoolean(false);
    private final AtomicInteger refCount;
    
    public CleanableTask(AtomicInteger refCount) {
        this.refCount = refCount;
    }
    
    public abstract void clean();
    
    @Override
    public final void run() {
        int count = refCount.decrementAndGet();
        if (count <= 0 && released.compareAndSet(false, true)) {
            clean();
        }
    }
}
