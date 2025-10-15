package org.brain4j.llm.core.model;

import org.brain4j.llm.api.ModelInfo;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public record LLM(
    String id,
    ModelInfo info,
    List<ModelFile> files,
    Map<String, Object> config
) {
    public Optional<ModelFile> find(String filename) {
        return files.stream().filter(file -> file.name().equals(filename)).findFirst();
    }

    public List<ModelFile> filesByFormat(String format) {
        return files.stream().filter(file -> file.format().equalsIgnoreCase(format)).toList();
    }

    public long totalSize() {
        return files.stream().mapToLong(ModelFile::size).sum();
    }
}