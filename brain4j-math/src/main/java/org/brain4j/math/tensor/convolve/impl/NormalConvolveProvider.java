package org.brain4j.math.tensor.convolve.impl;

import org.brain4j.math.tensor.convolve.ConvolveProvider;

public class NormalConvolveProvider implements ConvolveProvider {
    
    @Override
    public void dotPerFilter(int totalPatches, int patchSize, float[] filterData, int filterOffset, float[] patchData,
                             float[] outData, int outBase) {
        for (int p = 0; p < totalPatches; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;
            
            for (int i = 0; i < patchSize; i++) {
                sum += filterData[filterOffset + i] * patchData[rowOffset + i];
            }
            
            outData[outBase + p] = sum;
        }
    }
}
