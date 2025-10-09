package org.brain4j.backend.api.memory;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class TempObject<T> {
    
    public static final Cleaner CLEANER = Cleaner.create();
    
    private final AtomicInteger refCount = new AtomicInteger(1);

    private T value;
    
    public TempObject(T value, Runnable cleanerTask) {
        this.value = value;
        CLEANER.register(this, new CleanerTask(cleanerTask, refCount));
    }
    
    public AtomicInteger refCount() {
        return refCount;
    }
    
    public T value() {
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
        refCount.decrementAndGet();
    }

    static class CleanerTask implements Runnable {

        private final AtomicBoolean released = new AtomicBoolean(false);
        private final AtomicInteger refCount;
        private final Runnable cleanerTask;

        public CleanerTask(Runnable cleanerTask, AtomicInteger refCount) {
            this.cleanerTask = cleanerTask;
            this.refCount = refCount;
        }

        @Override
        public void run() {
            int count = refCount.decrementAndGet();
            if (count == 0 && released.compareAndSet(false, true)) {
                cleanerTask.run();
            }
        }
    }
}
