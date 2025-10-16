package org.brain4j.llm.core.model;

import java.util.Random;

public record SamplingConfig(Random random, int maxLength, int topK, double topP, double temperature) {
    
    public static SamplingConfig defaultConfig() {
        return new SamplingConfig(new Random(), 32, 5, 0.9, 1.0);
    }
    
    public static SamplingConfig.Builder builder() {
        return new SamplingConfig.Builder();
    }
    
    public static class Builder {

        private Random random = new Random();
        private int maxLength = -1;
        private int topK = 50;
        private double topP = 0.9;
        private double temperature = 1.0;

        public Builder setRandom(Random random) {
            this.random = random;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }
        
        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }
        
        public Builder topP(double topP) {
            this.topP = topP;
            return this;
        }
        
        public Builder temperature(double temperature) {
            this.temperature = temperature;
            return this;
        }
        
        public SamplingConfig build() {
            return new SamplingConfig(random, maxLength, topK, topP, temperature);
        }
    }
}
