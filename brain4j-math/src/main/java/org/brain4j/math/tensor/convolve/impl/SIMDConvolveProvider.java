package org.brain4j.math.tensor.convolve.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.math.tensor.convolve.ConvolveProvider;

public class SIMDConvolveProvider implements ConvolveProvider {
    
    @Override
    public void dotPerFilter(int totalPatches, int patchSize, float[] filterData, int filterOffset, float[] patchData,
                             float[] outData, int outBase) {
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        
        for (int p = 0; p < totalPatches; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;
            
            int i = 0;
            int loopBound = SPECIES.loopBound(patchSize);
            
            for (; i < loopBound; i += SPECIES.length()) {
                var v1 = FloatVector.fromArray(SPECIES, filterData, filterOffset + i);
                var v2 = FloatVector.fromArray(SPECIES, patchData, rowOffset + i);
                sum += v1.mul(v2).reduceLanes(VectorOperators.ADD);
            }
            
            for (; i < patchSize; i++) {
                sum += filterData[filterOffset + i] * patchData[rowOffset + i];
            }
            
            outData[outBase + p] = sum;
        }
    }
}
