package org.brain4j.core.monitor;

public interface Monitor {
    void batchCompleted(int batch);
    void epochCompleted(int epoch);
}
