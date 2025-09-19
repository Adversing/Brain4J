package org.brain4j.math.tensor.convolve;

public interface ConvolveProvider {
    void dotPerFilter(int totalPatches, int patchSize, float[] filterData, int filterOffset, float[] patchData,
                      float[] outData, int outBase);
}
