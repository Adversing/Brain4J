package org.brain4j.core.monitor;

public record DefaultMonitor(int evaluateEvery) implements Monitor {

    public DefaultMonitor() {
        this(-1);
    }

    @Override
    public void batchCompleted(int batch) {
    }

    @Override
    public void epochCompleted(int epoch) {
    }
}
