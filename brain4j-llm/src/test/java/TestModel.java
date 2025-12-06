import org.brain4j.llm.Models;
import org.brain4j.llm.core.model.LLM;
import org.brain4j.llm.core.model.SamplingConfig;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class TestModel {

    public static void main(String[] args) throws Exception {
        LLM llm = Models.loadModel("gpt2");
        llm.model().summary();
        String prompt = "Hello, my name is";

        AtomicLong lastToken = new AtomicLong(System.nanoTime());
        AtomicReference<Double> total = new AtomicReference<>(0.0);
        AtomicInteger generated = new AtomicInteger(0);

        System.out.print(prompt);
        SamplingConfig config = SamplingConfig.builder().maxLength(256).build();
        String response = llm.chat(prompt, config, token -> {
            long now = System.nanoTime();
            double took = (now - lastToken.get()) / 1e6;
            System.out.print(token);

            total.updateAndGet(v -> v + took);
            lastToken.set(now);
            generated.incrementAndGet();
        });
        double average = total.get() / generated.get();

        System.out.println("Total ms spent generating = " + total.get());
        System.out.println("Average ms/token = " + average);
    }
}
