package org.brain4j.llm.api;

public record SamplingConfig(double topK, double topP, double temperature) {
    
    public static SamplingConfig defaultConfig() {
        return new SamplingConfig(50, 0.9, 1.0);
    }
    
    public static SamplingConfig.Builder builder() {
        return new SamplingConfig.Builder();
    }
    
    public static class Builder {
        private int topK = 50;
        private double topP = 0.9;
        private double temperature = 1.0;
        
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
            return new SamplingConfig(topK, topP, temperature);
        }
    }
}
