package org.brain4j.math.tensor.parallel;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.convolve.ConvolveProvider;
import org.brain4j.math.tensor.convolve.impl.NormalConvolveProvider;
import org.brain4j.math.tensor.convolve.impl.SIMDConvolveProvider;
import org.brain4j.math.tensor.index.Range;

import java.util.stream.IntStream;

public class ParallelConvolve {

    public static Tensor convolve(Tensor a, Tensor b) {
        while (a.rank() < 4) a = a.unsqueeze();
        while (b.rank() < 4) b = b.unsqueeze();

        ConvolveProvider provider = DeviceUtils.isSimdAvailable()
            ? new SIMDConvolveProvider()
            : new NormalConvolveProvider();

        int[] aShape = a.shape();
        int[] bShape = b.shape();

        boolean aHasBatch = aShape.length == 4;
        int batch = aHasBatch ? aShape[0] : 1;
        int inChannels = aShape[aShape.length - 3];
        int inHeight = aShape[aShape.length - 2];
        int inWidth = aShape[aShape.length - 1];

        int numFilters = bShape[0];
        int filterHeight = bShape[2];
        int filterWidth = bShape[3];

        int outHeight = inHeight - filterHeight + 1;
        int outWidth = inWidth - filterWidth + 1;

        int patchSize = inChannels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        Tensor out = Tensors.zeros(batch, numFilters, outHeight, outWidth);
        Tensor filterFlat = b.reshape(numFilters, patchSize);

        float[] filterData = filterFlat.data();
        float[] outData = out.data();

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            Tensor inputBatch = (aHasBatch ? a.slice(Range.point(bIdx)) : a).squeeze(0);
            Tensor patchMatrix = Tensors.im2col(inputBatch, filterHeight, filterWidth);
            float[] patchData = patchMatrix.data();

            int finalBIdx = bIdx;
            IntStream.range(0, numFilters).parallel().forEach(f -> {
                int filterOffset = f * patchSize;
                int outBase = (finalBIdx * numFilters + f) * totalPatches;
                provider.dotPerFilter(totalPatches, patchSize,
                    filterData, filterOffset,
                    patchData, outData, outBase);
            });
        }

        return out;
    }
}