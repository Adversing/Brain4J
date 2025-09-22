package org.brain4j.math.tensor.parallel;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.convolve.ConvolveProvider;
import org.brain4j.math.tensor.convolve.impl.NormalConvolveProvider;
import org.brain4j.math.tensor.convolve.impl.SIMDConvolveProvider;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelConvolve extends RecursiveAction {

    private static final int PARALLEL_THRESHOLD = 1 << 12; // 4096 patches

    private record Im2ColParams(
        float[] inputData,
        float[] resultData,
        int inputBaseOffset,
        int resultBaseOffset,
        int channelCount,
        int inputHeight,
        int inputWidth,
        int filterHeight,
        int filterWidth,
        int outputHeight,
        int outputWidth
    ) {}

    private final Im2ColParams params;
    private final int startPatch;
    private final int endPatch;
    
    private ParallelConvolve(Im2ColParams params, int startPatch, int endPatch) {
        this.params = params;
        this.startPatch = startPatch;
        this.endPatch = endPatch;
    }

    @Override
    protected void compute() {
        int patchSize = params.channelCount * params.filterHeight * params.filterWidth;
        int work = endPatch - startPatch;

        if (work > PARALLEL_THRESHOLD) {
            int mid = (startPatch + endPatch) >>> 1;
            invokeAll(
                new ParallelConvolve(params, startPatch, mid),
                new ParallelConvolve(params, mid, endPatch)
            );
        } else {
            float[] inputData = params.inputData;
            float[] resultData = params.resultData;
            int inputBase = params.inputBaseOffset;
            int resultBase = params.resultBaseOffset;

            for (int patchIndex = startPatch; patchIndex < endPatch; patchIndex++) {
                int outRow = patchIndex / params.outputWidth;
                int outCol = patchIndex % params.outputWidth;
                int baseResultOffset = resultBase + patchIndex * patchSize;

                for (int c = 0; c < params.channelCount; c++) {
                    int channelOffsetInput = inputBase + c * params.inputHeight * params.inputWidth;
                    int channelOffsetResult = c * params.filterHeight * params.filterWidth;

                    for (int fh = 0; fh < params.filterHeight; fh++) {
                        int srcPos = channelOffsetInput + (outRow + fh) * params.inputWidth + outCol;
                        int destPos = baseResultOffset + channelOffsetResult + fh * params.filterWidth;
                        System.arraycopy(inputData, srcPos, resultData, destPos, params.filterWidth);
                    }
                }
            }
        }
    }

    public static Tensor convolve(Tensor a, Tensor b) {
        while (a.rank() < 3) a = a.unsqueeze();
        while (b.rank() < 3) b = b.unsqueeze();
        
        ConvolveProvider provider = DeviceUtils.isSimdAvailable() ? new SIMDConvolveProvider() : new NormalConvolveProvider();
        
        int[] aShape = a.shape();
        int[] bShape = b.shape();

        boolean aHasBatch = aShape.length == 4;
        int aBatch = aHasBatch ? aShape[0] : 1;
        int aChannels = aShape[aShape.length - 3];
        int aHeight = aShape[aShape.length - 2];
        int aWidth = aShape[aShape.length - 1];

        int numFilters;
        int bChannels;
        int filterHeight, filterWidth;

        if (bShape.length == 4) {
            numFilters = bShape[0];
            bChannels = bShape[1];
            filterHeight = bShape[2];
            filterWidth = bShape[3];
        } else {
            numFilters = 1;
            bChannels = bShape[0];
            filterHeight = bShape[1];
            filterWidth = bShape[2];
        }

        if (bChannels != aChannels) {
            throw new IllegalArgumentException("Channel mismatch: input channels=" + aChannels + " filter channels=" + bChannels);
        }

        int outHeight = aHeight - filterHeight + 1;
        int outWidth = aWidth - filterWidth + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw new IllegalArgumentException("Filter larger than input.");
        }

        int patchSize = aChannels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        Tensor out = Tensors.zeros(aBatch, numFilters, outHeight, outWidth);
        Tensor filterFlat = b.reshape(numFilters, patchSize);

        float[] aData = a.data();
        float[] filterData = filterFlat.data();
        float[] outData = out.data();

        for (int batchIdx = 0; batchIdx < aBatch; batchIdx++) {
            int inputBaseOffset = batchIdx * aChannels * aHeight * aWidth;
            float[] patchData = new float[totalPatches * patchSize];

            Im2ColParams params = new Im2ColParams(
                aData,
                patchData,
                inputBaseOffset,
                0,
                aChannels,
                aHeight,
                aWidth,
                filterHeight,
                filterWidth,
                outHeight,
                outWidth
            );

            try (ForkJoinPool pool = ForkJoinPool.commonPool()) {
                pool.invoke(new ParallelConvolve(params, 0, totalPatches));
            }

            for (int filter = 0; filter < numFilters; filter++) {
                int filterOffset = filter * patchSize;
                int outBase = aBatch > 1
                    ? (batchIdx * numFilters + filter) * totalPatches
                    : filter * totalPatches;
                
                provider.dotPerFilter(totalPatches, patchSize,
                    filterData, filterOffset,
                    patchData,
                    outData, outBase);
            }
        }
        
        return out.squeeze();
    }
}
