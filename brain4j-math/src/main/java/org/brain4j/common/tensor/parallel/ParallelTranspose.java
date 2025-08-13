package org.brain4j.common.tensor.parallel;

import org.brain4j.common.tensor.impl.BaseTensor;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ParallelTranspose extends RecursiveAction {
    
    public record TransposeParameters(float[] source, float[] dest, int rows, int cols, int planeSize) { }
    
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
            mapSection(parameters, start, end);
        }
    }
    
    public static void transpose(BaseTensor source, BaseTensor dest) {
        int[] shape = source.shape();
        int rank = shape.length;
        
        if (rank < 2) {
            throw new IllegalArgumentException("Source tensor must have rank >= 2 to transpose!");
        }
        
        int rows = shape[rank - 2];
        int cols = shape[rank - 1];
        
        int[] destShape = dest.shape();
        
        if (destShape.length != rank || destShape[rank - 2] != cols || destShape[rank - 1] != rows) {
            throw new IllegalArgumentException("Dest must have swapped last two dimensions");
        }
        
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= shape[i];
        
        int planeSize = rows * cols;
        int totalElements = batch * planeSize;
        
        TransposeParameters params = new TransposeParameters(source.data(), dest.data(), rows, cols, planeSize);
        
        if (!isOverParallelThreshold(totalElements)) {
            mapSection(params, 0, totalElements);
            return;
        }
        
        int step = totalElements / PARALLELISM;
        ParallelTranspose[] tasks = new ParallelTranspose[PARALLELISM];
        
        for (int i = 0; i < PARALLELISM; i++) {
            int start = i * step;
            int end = Math.min(start + step, totalElements);
            tasks[i] = new ParallelTranspose(params, start, end);
        }
        
        ForkJoinTask.invokeAll(tasks);
    }
    
    private static void mapSection(TransposeParameters p, int start, int end) {
        float[] src = p.source();
        float[] dst = p.dest();
        int rows = p.rows();
        int cols = p.cols();
        int planeSize = p.planeSize();
        
        for (int lin = start; lin < end; lin++) {
            int b = lin / planeSize;
            int offset = lin - b * planeSize;
            int i = offset / cols;
            int j = offset - i * cols;
            
            int srcIdx = b * planeSize + i * cols + j;
            int dstIdx = b * planeSize + j * rows + i;
            
            dst[dstIdx] = src[srcIdx];
        }
    }
}
