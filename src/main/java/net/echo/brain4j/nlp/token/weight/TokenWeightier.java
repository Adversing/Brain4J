package net.echo.brain4j.nlp.token.weight;

import java.util.HashMap;
import java.util.Map;

public class TokenWeightier {

    private final Map<String, Double> vocabulary;
    private final double smoothingFactor;

    public TokenWeightier(double smoothingFactor) {
        this.vocabulary = new HashMap<>();
        this.smoothingFactor = smoothingFactor;
    }

    // TODO
    public void updateWeights(String token, double frequency) {
        double idf = Math.log(1 + (1.0 / (frequency + smoothingFactor)));
        vocabulary.put(token, idf);
    }

    public double getWeight(String token) {
        return vocabulary.getOrDefault(token, 1.0);
    }
}