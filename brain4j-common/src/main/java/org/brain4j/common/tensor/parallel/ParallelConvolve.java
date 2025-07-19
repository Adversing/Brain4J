package org.brain4j.common.tensor.parallel;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.common.Tensors;
import org.brain4j.common.gpu.device.DeviceUtils;
import org.brain4j.common.tensor.Tensor;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelConvolve extends RecursiveAction {

    private static final int PARALLEL_THRESHOLD = 1 << 12; // 4096 patches

    private record Im2ColParams(
        float[] inputData,
        float[] resultData,
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

            for (int patchIndex = startPatch; patchIndex < endPatch; patchIndex++) {
                int outRow = patchIndex / params.outputWidth;
                int outCol = patchIndex % params.outputWidth;
                int baseResultOffset = patchIndex * patchSize;

                for (int c = 0; c < params.channelCount; c++) {
                    int channelOffsetInput = c * params.inputHeight * params.inputWidth;
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
        while (a.rank() < 3) a.unsqueeze();
        while (b.rank() < 3) b.unsqueeze();

        int[] inShape = a.shape();
        int[] fShape = b.shape();

        int channelCount = inShape[inShape.length - 3];
        int inputHeight = inShape[inShape.length - 2];
        int inputWidth = inShape[inShape.length - 1];
        int filterHeight = fShape[fShape.length - 2];
        int filterWidth = fShape[fShape.length - 1];

        int outputHeight = inputHeight - filterHeight + 1;
        int outputWidth = inputWidth - filterWidth + 1;

        int patchSize = channelCount * filterHeight * filterWidth;
        int totalPatches = outputHeight * outputWidth;

        Tensor patchMatrix = Tensors.zeros(totalPatches, patchSize);
        float[] inputData = a.data();
        float[] patchData = patchMatrix.data();

        Im2ColParams params = new Im2ColParams(
            inputData,
            patchData,
            channelCount,
            inputHeight,
            inputWidth,
            filterHeight,
            filterWidth,
            outputHeight,
            outputWidth
        );

        try (ForkJoinPool pool = ForkJoinPool.commonPool()) {
            pool.invoke(
                new ParallelConvolve(params, 0, totalPatches)
            );
        }

        Tensor filterFlat = b.reshape(1, b.elements());
        Tensor outputTensor = Tensors.zeros(outputHeight, outputWidth);

        float[] filterData = filterFlat.data();
        float[] outputData = outputTensor.data();

        if (DeviceUtils.isSimdAvailable()) {
            simdDotProduct(totalPatches, patchSize, filterData, patchData, outputData);
        } else {
            normalDotProduct(totalPatches, patchSize, filterData, patchData, outputData);
        }

        return outputTensor;
    }

    private static void normalDotProduct(
        int totalPatches,
        int patchSize,
        float[] filterData,
        float[] patchData,
        float[] outputData
    ) {

        for (int p = 0; p < totalPatches; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;

            for (int i = 0; i < patchSize; i++) {
                sum += filterData[i] * patchData[rowOffset + i];
            }

            outputData[p] = sum;
        }
    }

    private static void simdDotProduct(
        int totalPatches,
        int patchSize,
        float[] filterData,
        float[] patchData,
        float[] outputData
    ) {
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

        for (int p = 0; p < totalPatches; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;

            int i = 0;

            for (; i < SPECIES.loopBound(patchSize); i += SPECIES.length()) {
                var v1 = FloatVector.fromArray(SPECIES, filterData, i);
                var v2 = FloatVector.fromArray(SPECIES, patchData, rowOffset + i);
                sum += v1.mul(v2).reduceLanes(VectorOperators.ADD);
            }

            for (; i < patchSize; i++) {
                sum += filterData[i] * patchData[rowOffset + i];
            }

            outputData[p] = sum;
        }
    }
}
