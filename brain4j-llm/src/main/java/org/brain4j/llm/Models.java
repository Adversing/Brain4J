package org.brain4j.llm;

import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.llm.core.loader.ModelLoader;
import org.brain4j.llm.core.loader.config.LoadConfig;
import org.brain4j.llm.core.model.LLM;

public class Models {

    public static LLM loadModel(String modelId) throws Exception {
        try (ModelLoader loader = new ModelLoader()) {
            return loader.loadModel(modelId, LoadConfig.defaultConfig());
        } catch (Exception e) {
            throw new Exception("Failed to load model: " + modelId, e);
        }
    }
    
    public static Tokenizer loadTokenizer(String tokenizerId) throws Exception {
        try (ModelLoader loader = new ModelLoader()) {
            return loader.loadTokenizer(tokenizerId, LoadConfig.defaultConfig());
        } catch (Exception e) {
            throw new Exception("Failed to load tokenizer: " + tokenizerId, e);
        }
    }
}
