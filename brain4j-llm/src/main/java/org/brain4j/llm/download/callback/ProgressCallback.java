package org.brain4j.llm.download.callback;

@FunctionalInterface
public interface ProgressCallback {

    void onProgress(String filename, double percentage, String message);

    /**
     * No-operation progress callback.
     */
    ProgressCallback NOOP = (filename, percentage, message) -> {};
}