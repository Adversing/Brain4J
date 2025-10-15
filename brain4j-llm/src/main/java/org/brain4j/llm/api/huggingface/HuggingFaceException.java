package org.brain4j.llm.api.huggingface;

public class HuggingFaceException extends Exception {
    public HuggingFaceException(String msg) {
        super(msg);
    }

    public HuggingFaceException(String msg, Throwable t) { super
        (msg, t);
    }
}