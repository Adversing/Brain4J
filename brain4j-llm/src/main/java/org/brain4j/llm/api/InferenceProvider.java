package org.brain4j.llm.api;

public interface InferenceProvider {
    String chat(String prompt);
    String chat(String prompt, SamplingConfig config);
}
