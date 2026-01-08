package org.brain4j.math.gpu;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class TempObject<T> {
    
    public static final Cleaner CLEANER = Cleaner.create();
    
    protected final AtomicInteger refCount = new AtomicInteger(1);
    private CleanableTask cleanerTask;
    private T value;
    
    public TempObject(T value) {
        this.value = value;
    }
    
    public void register(CleanableTask task) {
        this.cleanerTask = task;
        CLEANER.register(this, task);
    }
    
    public AtomicInteger refCount() {
        return refCount;
    }
    
    public T getValue() {
        return value;
    }
    
    public T setValue(T value) {
        this.value = value;
        return value;
    }
    
    public void retain() {
        refCount.incrementAndGet();
    }
    
    public void release() {
        release(false);
    }
    
    public void release(boolean instant) {
        if (instant) {
            cleanerTask.run();
        } else {
            GpuContext.RELEASE_QUEUE.add(cleanerTask);
        }
    }
}
