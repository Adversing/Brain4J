package org.brain4j.llm.core.model;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import org.brain4j.core.importing.SafeTensorsConverter;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.core.transformer.tokenizers.impl.BytePairTokenizer;
import org.brain4j.llm.api.ModelFile;
import org.brain4j.llm.api.ModelInfo;
import org.brain4j.llm.core.architecture.ArchitectureAdapter;
import org.brain4j.llm.core.architecture.ArchitectureRegistry;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
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
        this.tokenizer = new BytePairTokenizer();
        
        ModelFile configFile = findOrThrow("config.json", "config.json was not found!");
        ModelFile weightsFile = findOrThrow("model.safetensors", "model.safetensors was not found!");
        ModelFile tokenizerFile = findOrThrow("tokenizer.json", "tokenizer.json was not found!");
        
        String configText = new String(Files.readAllBytes(configFile.path()));
        JsonObject config = GSON.fromJson(configText, JsonObject.class);
        String modelType = config.get("model_type").getAsString();
        
        tokenizer.load(tokenizerFile.path().toFile());
        tokenizer.setBosTokenId(config.get("bos_token_id").getAsInt());
        tokenizer.setEosTokenId(config.get("eos_token_id").getAsInt());
        
        byte[] modelWeights = Files.readAllBytes(weightsFile.path());
        Map<String, Tensor> weights = SafeTensorsConverter.load(modelWeights);
        
        ArchitectureAdapter adapter = ArchitectureRegistry.findAdapter(modelType);
        this.model = adapter.buildModel(config, weights);
    }
    
    @Override
    public String chat(String prompt) {
        return chat(prompt, SamplingConfig.defaultConfig());
    }
    
    @Override
    public String chat(String prompt, SamplingConfig config) {
        List<String> tokens = tokenizer.splitTokens(prompt);
        Tensor input = tokenizer.encode(tokens);

        StatesCache cache = new StatesCache();
        StringBuilder response = new StringBuilder();
        
        int bosToken = tokenizer.bosTokenId();
        int eosToken = tokenizer.eosTokenId();
        int generatedTokens = 0;
        
        input = input.concat(Tensors.scalar(bosToken));
        
        while (generatedTokens < config.maxLength()) {
            Tensor distribution = model.predict(cache, input.squeeze())[0]; // [1, seq_len, vocab]
            int seqLen = distribution.shape(1);
            
            Range[] ranges = { Range.all(), Range.point(seqLen - 2), Range.all() };
            Tensor lastToken = distribution.slice(ranges).squeeze(); // [vocab]
            
            int index = lastToken.argmax(); // TODO: top-p/top-k sampling
            input = input.concat(Tensors.scalar(index));
            response.append(tokenizer.decode(index)).append(" ");
            
            if (index == eosToken) break;
            
            generatedTokens++;
        }
        
        return response.toString();
    }
    
    public Optional<ModelFile> find(String filename) {
        return files.stream().filter(file -> file.name().equals(filename)).findFirst();
    }
    
    private ModelFile findOrThrow(String filename, String message) throws FileNotFoundException {
        return find(filename).orElseThrow(() -> new FileNotFoundException(message));
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