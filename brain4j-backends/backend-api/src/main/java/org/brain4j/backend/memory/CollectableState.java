package org.brain4j.backend.memory;

public interface CollectableState extends Runnable {
    void release(MemoryObject object);
}
