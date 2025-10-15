import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.llm.Models;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class TestModel {

    public static void main(String[] args) throws Exception {
        LLM llm = Models.loadModel("gpt2");
        String prompt = "When I realized that AI could write poetry, I questioned what creativity really means.";
        
        llm.compile();
        
        System.out.println("LLM ID: " + llm.id());
        System.out.println("Total bytes: " + Commons.formatNumber(llm.totalSize() / 64));
        
        
        Tokenizer tokenizer = Models.loadTokenizer("gpt2");
        System.out.println("Loaded BPE tokenizer from GPT2: " + tokenizer.getClass().getSimpleName());
        
        List<String> splitted = tokenizer.splitTokens(prompt);
        Tensor out = tokenizer.encode(splitted);
        System.out.println("Prompt: " + prompt);
        System.out.println("Splitted tokens: " + splitted);
        System.out.println("LLM Input: " + out.toString("%.0f"));
    }
}
