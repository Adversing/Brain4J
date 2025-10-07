package org.brain4j.math.gpu;

import org.jocl.CL;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class TempObject<T> implements Runnable {
    
    public static final Cleaner CLEANER = Cleaner.create();
    
    private final AtomicInteger refCount = new AtomicInteger(1);
    private final AtomicBoolean released = new AtomicBoolean(false);
    private final Cleaner.Cleanable cleanable;
    private final Runnable cleanerTask;
    
    private T value;
    
    public TempObject(T value, Runnable cleanerTask) {
        this.value = value;
        this.cleanable = CLEANER.register(this, this);
        this.cleanerTask = cleanerTask;
    }
    
    public AtomicInteger refCount() {
        return refCount;
    }
    
    public T value() {
        return value;
    }
    
    public void setValue(T value) {
        this.value = value;
    }
    
    public void retain() {
        refCount.incrementAndGet();
    }
    
    public void release() {
        if (refCount.decrementAndGet() == 0) {
            cleanable.clean();
        }
    }
    
    @Override
    public void run() {
        refCount.decrementAndGet();
        if (refCount.get() == 0 && released.compareAndSet(false, true)) {
            cleanerTask.run();
        }
    }
}
