package org.brain4j.math.tensor.parallel;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.impl.BaseTensor;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ParallelTranspose extends RecursiveAction {
    
    public record TransposeParameters(float[] srcData, float[] dstData, int[] shape, int[] newShape, int dim1, int dim2) { }
    
    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 1 << 12; // 4096
    private static final int SPLIT_COMPLEXITY_THRESHOLD = 1 << 10; // 1024
    
    private static boolean isOverParallelThreshold(int work) {
        return work > PARALLEL_WORK_THRESHOLD && work > PARALLEL_COMPLEXITY_THRESHOLD;
    }
    
    private static boolean isOverSplitThreshold(int work) {
        return work > SPLIT_COMPLEXITY_THRESHOLD;
    }
    
    private final TransposeParameters parameters;
    private final int start;
    private final int end;

    public ParallelTranspose(TransposeParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end  = end;
    }
    
    @Override
    protected void compute() {
        int work = end - start;
        if (isOverSplitThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                new ParallelTranspose(parameters, start, mid),
                new ParallelTranspose(parameters, mid, end)
            );
        } else {
            mapSection(parameters.srcData, parameters.dstData, parameters.shape, parameters.newShape, parameters.dim1,
                parameters.dim2, start, end);
        }
    }

    public static void transpose(BaseTensor source, BaseTensor result, int dim1, int dim2) {
        // TODO
    }

    private static void mapSection(float[] src, float[] dst, int[] shape, int[] newShape,
                                   int dim1, int dim2, int start, int end) {
        int rank = shape.length;
        int[] newCoords = new int[rank];

        for (int lin = start; lin < end; lin++) {
            int[] coords = Tensors.unravelIndex(lin, shape);

            System.arraycopy(coords, 0, newCoords, 0, rank);
            int t = newCoords[dim1];
            newCoords[dim1] = newCoords[dim2];
            newCoords[dim2] = t;

            int dstIdx = 0;
            int stride = 1;
            for (int k = rank - 1; k >= 0; k--) {
                dstIdx += newCoords[k] * stride;
                stride *= newShape[k];
            }

            dst[dstIdx] = src[lin];
        }
    }
}
