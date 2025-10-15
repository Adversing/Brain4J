package org.brain4j.llm.core.model;

public interface InferenceProvider {
    String chat(String prompt);
    String chat(String prompt, SamplingConfig config);
}
