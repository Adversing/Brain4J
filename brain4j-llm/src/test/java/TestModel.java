import org.brain4j.core.transformer.tokenizers.Tokenizer;
import org.brain4j.llm.Models;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.llm.core.model.SamplingConfig;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

import java.nio.ByteOrder;
import java.util.List;

public class TestModel {

    public static void main(String[] args) throws Exception {
        LLM llm = Models.loadModel("gpt2-xl");
        llm.model().summary();
        String prompt = "Hello, my name is";
        System.out.print(prompt);
        String response = llm.chat(prompt, SamplingConfig.defaultConfig(), System.out::print);
    }
}
