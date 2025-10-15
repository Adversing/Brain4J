package org.brain4j.core.transformer.tokenizers;

import org.brain4j.math.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface Tokenizer {
    
    List<String> splitTokens(String input);
    Tensor encode(List<String> tokens);
    String decode(int index);
    
    Map<String, Integer> vocab();
    int vocabSize();
    
    void save(File file) throws IOException;
    void load(File file) throws IOException;
}
