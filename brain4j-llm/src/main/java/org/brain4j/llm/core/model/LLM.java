package org.brain4j.llm.core.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.core.transformer.tokenizers.impl.BytePairTokenizer;
import org.brain4j.llm.api.InferenceProvider;
import org.brain4j.llm.api.ModelInfo;
import org.brain4j.llm.api.SamplingConfig;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public class LLM implements InferenceProvider {
    
    public static Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    
    private final String id;
    private final ModelInfo info;
    private final List<ModelFile> files;
    private final Map<String, Object> config;
    
    // Inference
    private Model model;
    private Tokenizer tokenizer;
    
    public LLM(String id, ModelInfo info, List<ModelFile> files, Map<String, Object> config) {
        this.id = id;
        this.info = info;
        this.files = files;
        this.config = config;
    }
    
    public void compile() throws IOException {
        this.model = Sequential.of();
        
        Optional<ModelFile> optConfigFile = find("config.json");
        Optional<ModelFile> optWeightsFile = find("model.safetensors");
        Optional<ModelFile> optTokenizerFile = find("tokenizer.json");
        
        if (optConfigFile.isEmpty()) {
            throw new FileNotFoundException("config.json was not found!");
        }
        
        if (optWeightsFile.isEmpty()) {
            throw new FileNotFoundException("model.safetensors was not found!");
        }
        
        if (optTokenizerFile.isEmpty()) {
            throw new FileNotFoundException("tokenizer.json was not found!");
        }
        
        ModelFile tokenizerFile = optTokenizerFile.get();
        this.tokenizer = new BytePairTokenizer();
        this.tokenizer.load(tokenizerFile.path().toFile());
        System.out.println("Everything found!");
    }
    
    @Override
    public String chat(String prompt) {
        return chat(prompt, SamplingConfig.defaultConfig());
    }
    
    @Override
    public String chat(String prompt, SamplingConfig config) {
        return "";
    }
    
    public Optional<ModelFile> find(String filename) {
        return files.stream().filter(file -> file.name().equals(filename)).findFirst();
    }
    
    public List<ModelFile> filesByFormat(String format) {
        return files.stream().filter(file -> file.format().equalsIgnoreCase(format)).toList();
    }
    
    public long totalSize() {
        return files.stream().mapToLong(ModelFile::size).sum();
    }
    
    public String id() {
        return id;
    }
    
    public ModelInfo info() {
        return info;
    }
    
    public List<ModelFile> files() {
        return files;
    }
    
    public Map<String, Object> config() {
        return config;
    }
    
    public Model model() {
        return model;
    }
    
    public Tokenizer tokenizer() {
        return tokenizer;
    }
}