import org.brain4j.core.Brain4J;
import org.brain4j.llm.Models;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.llm.core.model.SamplingConfig;
import org.brain4j.math.gpu.device.Device;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

public class TestModel {

    public static void main(String[] args) throws Exception {
        LLM llm = Models.loadModel("gpt2");

        SamplingConfig config = SamplingConfig.builder().maxLength(256).build();
        TokenHandler handler = new TokenHandler();
        String prompt = "Hello, my name is";

        Device device = Brain4J.firstDevice();
        if (device != null) {
            System.out.printf("Using device %s %n", device.name());
            llm.move(device);
        }
        
        llm.getModel().summary();
        llm.chat(prompt, config, handler);
        handler.printStats();
    }
    
    private static class TokenHandler implements Consumer<String> {
        
        private long lastTokenTime;
        private double totalTime;
        private int generatedTokens;
        
        public TokenHandler() {
            this.lastTokenTime = System.nanoTime();
        }
        
        @Override
        public void accept(String s) {
            long now = System.nanoTime();
            double took = (now - lastTokenTime) / 1e6;
            System.out.print(s);
            
            this.lastTokenTime = now;
            this.totalTime += took;
            this.generatedTokens++;
        }
        
        public void printStats() {
            double average = totalTime / generatedTokens;
            
            System.out.println();
            System.out.printf("%s generated tokens %n", generatedTokens);
            System.out.printf("total ms  = %.2f %n", totalTime);
            System.out.printf("avg/token = %.2f %n", average);
        }
    }
}
